#Author : Karl Marrett kdmarrett@gmail.com

import numpy as np
import numpy.linalg as LA
import warnings
import cvxpy as cv
import matplotlib.pyplot as plt
import math
import scipy.integrate as sio

class DMD:
    
    def __init__(self, dt=1, r=1e32, scale_modes=True, stack_factor='estimate',
            use_optimal_SVHT=False, jovanovich=False,
            condensed_jovanovich=False):
        """
        Dynamic Mode Decomposition (DMD)  
        
        Estimates the modes of the dynamics of matrix X.  Each spatial mode has
        corresponding growth and frequency characteristics.

        Parameters
        ----------
        dt : float
            Timestep of data
        r : int
            Number of modes to truncate to 
        scale_modes : boolean
            Scale the spatial modes
        stack_factor : int, [string]
            The number of times to stack the X matrix upon `fit` such that
            the train matrix has more rows than columns.
        use_optimal_SVHT : boolean
            Use optimal SVHT to estimate the number of modes to truncate to i.e. `self.r`
        jovanovich : boolean
            Deprecated
        condensed_jovanonich : boolean
            Deprecated

        Returns
        --------
        mrDMD : object
            The blank instance of mrDMD with given parameters.

        See Also
        --------
        class: `mrDMD` 

        """
        self.jovanovich = jovanovich
        self.condensed_jovanovich = condensed_jovanovich
        self.Vand = None
        self.alphas = None
        self.real = True # assume X and Xhat to be real
        self.Phi = None # the modes
        #denoted as omega 
        self.mu = None #fourier spectrum of modes (mu = log(lambda)/dt)
        self.timesteps = None #the default # of timesteps
        self.lambdas = None #D, the DMD spectrum of modes 
        self.diagS = None # the singular values of the data matrix
        self.x0 = None #initial condition vector corresponding to Phi
        self.dt = dt # timestep
        self.r = r  # number to truncate DMD modes to
        self.scale_modes = scale_modes
        self.Xraw = None
        self.Xhat = None
        self.z0 = None
        self.Atilde = None
        self.stack_factor = stack_factor
        self.Xaug = None
        self.use_optimal_SVHT = use_optimal_SVHT

    def _augment_x(self, Xraw):
        """ Stack the features of the data Xraw such that 
        timesteps >= features where the rows of Xraw is the 
        number of features/channels
        and the columns of Xraw are the timesteps.

        Parameters
        --------
        Xraw : matrix-like
            The raw data matrix.

        Returns
        --------
        Xaug : matrix-like
            The augmented raw data matrix.
        """

        shape = Xraw.shape
        #estimate optimal stacking
        if self.stack_factor == 'estimate':
            self._estimate_stack_factor(*shape)
            assert(self.stack_factor != 'estimate')
        else:
            if self.stack_factor < 0:
                raise ValueError("`stack_factor` can not be negative")
            if type(self.stack_factor) != int:
                raise ValueError("`stack_factor` must be of type `int`")
            if self.stack_factor < 2: #if user inputted stack_factor of 1
                warnings.warn('stack_factor must always be at least 2 or greater to capture frequency content')
                self.Xaug = Xraw
                return Xraw
            else:
                # Xaug can not be less than 2 columns 
                if (shape[1] - (self.stack_factor - 1)) < 2: 
                    raise ValueError("`stack_factor` can not exceed X.shape[1] due to shifting")
        new_col = shape[1] - (self.stack_factor - 1)
        if new_col < 2:
            raise ValueError("`timesteps` are too low for given `stack_factor`")
        #concatenate by shift stacking features
        row_block = shape[0] * self.stack_factor
        Xaug = np.full((row_block, new_col), np.nan)
        Xaug[:shape[0], :] = Xraw[:, :new_col]
        for i in range(1, self.stack_factor):
            start = i * shape[0]
            Xaug[(start):(start + shape[0]), :] = Xraw[:, i:(new_col + i)]
        self.Xaug = Xaug
        return Xaug


    def _truncate(self, raw_shape, U, S, V, diagS):
        """
        Handle the truncation of the SVD decomposition,
        either by truncating to the prespecified r inputed
        during initialization or by calling the optimal
        hard threshold.

        Parameters
        ----------
        raw_shape : tuple
            The shape of Xaug or Xraw
        U : matrix-like
        S : matrix-like
        V : matrix-like
        diagS : array-like
            Diagonal values of S

        Returns
        -------
        U : matrix-like truncated to r columns
        S : matrix-like truncated to r columns
        V : matrix-like truncated to r columns

        See Also
        --------
        :class `DMD._estimate_optimal_SVHT`

        """
        if len(diagS.shape) != 1:
            raise ValueError("`diagS` must be array-like")
        if self.use_optimal_SVHT:
            self._estimate_optimal_SVHT(raw_shape, diagS)
        if U.shape[1] <= self.r:
            return U, S, V
        S = S[:self.r, :self.r]
        self.diagS = diagS[:self.r]
        U = U[:,:self.r]
        V = V[:,:self.r]
        return U, S, V

    def fit(self, Xraw):
        """ 
        Public call to fit DMD modes to Xraw

        Parameters
        ----------
        Xraw : matrix-like
            The raw data matrix

        Returns
        -------
        self : DMD object
            Returns the instance of itself

        See Also
        --------
        :class:`DMD._fit` : private call on `fit`

        """
        self._fit(Xraw)
        return self

    def _fit(self, Xraw):
        """ Private call to fit DMD modes to Xraw

        Parameters
        ----------
        Xraw : matrix-like
            The raw data matrix

        See Also
        --------
        :class:`DMD.fit` : public call on `_fit`

        """
        if isinstance(Xraw, complex): self.real = False
        raw_shape = Xraw.shape
        assert(len(raw_shape) == 2)
        if self.timesteps is None:
            self.timesteps = raw_shape[1]
        self.Xraw = Xraw.copy()
        Xraw = self._augment_x(Xraw)
        self.x0 = Xraw[:,0].copy()
        X = Xraw[:,:-1].copy()
        Y = Xraw[:,1:].copy()
        #compute the 'econ' matrix of X1
        [U, diagS, V_t] = LA.svd(X, full_matrices=False)
        #!! is the transpose (different from Matlab)
        V = V_t.T
        if all(v==0 for v in diagS):
            warnings.warn('Xraw is a 0 vector')
        S = np.diag(diagS)
        U_r, S_r, V_r = self._truncate(raw_shape, U, S, V, diagS)
        Atilde = U_r.T.dot(Y).dot(V_r).dot(np.diag(1 / np.diag(S_r)))
        assert( not np.isinf(Atilde).any())
        if self.scale_modes and (not self.jovanovich) \
            and (not self.condensed_jovanovich): # scaling modes
            S_r_neg = S_r.copy() # S^-1/2
            S_r_pow = S_r.copy() # S^1/2
            S_r_neg[np.diag_indices(len(S_r))] = 1 / (S_r_neg.diagonal()**0.5)
            S_r_pow[np.diag_indices(len(S_r))] = S_r_pow.diagonal()**0.5
            Ahat = np.dot(S_r_neg, Atilde).dot(S_r_pow)
            #below: theoretic. equivalent but impossible in python:
            #Ahat = (S_r^(-1/2)).dot(Atilde).dot(S_r^(1/2))
            lambdas, What = LA.eig(Ahat)
            W = S_r_pow.dot(What)
        else:
            # W is the matrix of eigen vectors
            #lambdas is the DMD eigen values
            lambdas, W = LA.eig(Atilde)
        if self.jovanovich or self.condensed_jovanovich:
            Phi = U_r.dot(W) # alternate calculation of Phi
            Vand = np.vander(lambdas, raw_shape[1], increasing=True)
            self.Vand = Vand.copy()
            Vand = Vand[:, :X.shape[1]]
            d = cv.Variable(len(lambdas))
            if self.condensed_jovanovich:
                #match the dimensions of S since Y is stacked anyway
                if W.shape[0] > S.shape[0]: 
                    local_W = W[:S.shape[0],:]
                else:
                    local_W = W
                SV = S.dot(V_t)
                objective = cv.Minimize(cv.square(cv.norm(SV
                    - local_W * cv.diag(d) * Vand, "fro")))
            else:
                objective = cv.Minimize(cv.square(cv.norm(X 
                    - Phi * cv.diag(d) * Vand, "fro")))
            constraints = [d >= 0.0]
            #import pdb; pdb.set_trace()
            prob = cv.Problem(objective, constraints)
            optimal_value = prob.solve()
            self.alphas = np.array(d.value)
            #TODO add in constraints of power list of bools involving d
            #TODO add additional method using E V*
        else:
            Phi = Y.dot(V_r).dot(np.diag(1 / np.diag(S_r))).dot(W)
        self.Phi = Phi
        self.lambdas = lambdas
        self.Atilde = Atilde
        if not any(self.lambdas.imag):
            warnings.warn("Lambdas contain no complex components, self.r : %d" % self.r)
        #np.log accepts negative complex values
        self.mu = np.log(lambdas) / self.dt #denoted as omega in paper

    def fit_transform(self, Xraw, timesteps='default', compute_error=False,
        keep_modes=None, unaugment=True):
        """ 
        Fits the DMD modes to the data and creates a reconstructed
        data matrix Xhat.  Also updates the reconstruction error.

        Parameters
        --------
        Xraw : matrix-like
            Raw data matrix
        timesteps : float
            Number of timesteps to include in the reconstructed data
            matrix.  If timesteps == 'default', it will use the original columns
            of the Xraw matrix passed in.
        compute_error : Boolean
            If true returns the reconstruction error : |Xraw - Xhat|
        keep_modes : array-like
            An array of indices to the modes (columns) to keep in the reconstruction
            Default is None which uses all modes of Phi to reconstruct
        unaugment : Boolean
            Augment the Xraw via shift stacking.  See self._estimate_stack_factor
            and cited paper for discussion on this behavior. 

        Returns
        --------
        Xhat : matrix-like
            The reconstructed Xaug
        E : scalar
            The reconstruction error

        See Also
        --------
        :class: `DMD.transform` : public call on `transform`
        :class:`DMD.fit` : public call on `fit`
        """
        if timesteps is 'default':
            timesteps = self.Xraw.shape[1]
        self._fit(Xraw)
        self.timesteps = timesteps
        self.keep_modes = keep_modes
        Xhat = self._transform(keep_modes,
                compute_error=compute_error, unaugment=unaugment)
        if compute_error:
            return Xhat, self.E
        else:
            return Xhat

    def transform(self, timesteps='default', compute_error=False,
            keep_modes=None, unaugment=True):
        """
        Public call on _transform.
        Reconstructs the original data matrix Xaug
        from the DMD modes and initial conditions

        Parameters
        --------
        timesteps : float
            number of timesteps to include in the reconstructed data
            matrix
        compute_error : boolean
            If true returns the reconstruction error : |Xraw - Xhat|
        keep_modes : array-like
            An array of indices to the modes (columns) to keep in the reconstruction
            Default is None which uses all modes of Phi to reconstruct Xhat
        unaugment : boolean
            augment the Xraw via shift stacking.  See self._estimate_stack_factor
            and cited paper for discussion on this behavior. 

        Returns
        --------
        Xhat : matrix-like, float
            The reconstructed Xaug
        E : scalar
            The reconstruction error

        See Also
        --------
        :class: `DMD._transform` : private call on `transform`

        """
        if self.Xraw is None:
            raise ValueError('Xraw is None, you must call fit()\
                    or fit_transform() before calling\
                    transform()')
        if timesteps is 'default':
            timesteps = self.Xraw.shape[1]
        self.timesteps = timesteps
        self.keep_modes = keep_modes
        Xhat = self._transform(keep_modes,
                compute_error=compute_error, unaugment=unaugment)
        if compute_error:
            return Xhat, self.E
        else:
            return Xhat

    def _transform(self, keep_modes, compute_error=False, unaugment=True, t_list='default'):
        """
        Reconstruct the original data matrix Xaug
        from the DMD modes and initial conditions.

        Parameters
        ----------
        keep_modes : array-like
            An array of indices to the modes (columns) to keep in the reconstruction
            Default is None which uses all modes of Phi to reconstruct Xhat
        compute_error : boolean
            If true returns the reconstruction error : |Xraw - Xhat|
        unaugment : boolean
            Augment the Xraw via shift stacking.  See self._estimate_stack_factor
            and cited paper for discussion on this behavior. 
        t_list : array-like
            Create reconstruction for custom list of times

        Returns
        -------
        Xhat : matrix-like, float, (features, timesteps)
            The reconstructed Xaug where timesteps is the length of x0
        E : scalar
            The reconstruction error

        Notes
        -----
        Xhat will only come out with non-zero imaginary
        components; if the original data matrix Xraw was not
        strictly real valued, otherwise Xhat will also be a
        complex matrix.

        See Also
        --------
        :class: `DMD._transform` : private call on `transform`

        """
        if t_list is 'default':
            timesteps = self.timesteps
        else:
            timesteps = len(t_list)
        Phi = self.Phi
        Vand = self.Vand
        alphas = self.alphas
        lambdas = self.lambdas
        #update mu in case dt has changed
        mu = np.log(lambdas) / self.dt #denoted as omega in paper
        alphas = np.squeeze(self.alphas)
        #when keep_modes is not None, truncate to those modes
        if keep_modes:
            Phi = Phi[:, keep_modes]
            if self.jovanovich or self.condensed_jovanovich:
                #truncate
                Vand = Vand[keep_modes, :]
                alphas = alphas[keep_modes]
            else:
                #truncate
                mu = mu[keep_modes]
                lambdas = lambdas[keep_modes]

        if self.jovanovich or self.condensed_jovanovich:
            Xhat = Phi.dot(np.diag(alphas)).dot(Vand)
        else:
            #pseudo inverse to find initial conditions
            self.z0 = LA.pinv(Phi).dot(self.x0)
            if self.real:
                #if X was real, cast to real
                Z = np.full((len(self.z0), timesteps), np.nan)
            else:
                #if raw matrix contained complex 
                Z = np.full((len(self.z0), timesteps), np.nan,
                        dtype=np.complex)
            if t_list is 'default':
                for ti in range(self.timesteps):
                    #Z[:, ti] = self.z0 * (lambdas ** (ti + 1))
                    Z[:, ti] = self.z0 * np.exp(mu * (self.dt * (ti + 1)))
            else:
                for ti in t_list:
                    #Z[:, ti] = self.z0 * (lambdas ** (ti + 1))
                    Z[:, ti] = self.z0 * np.exp(mu * (ti + 1))
            Xhat = Phi.dot(Z)
            self.Z = Z
        if self.real:
            Xhat = Xhat.real
        self.Xhat = Xhat
        if compute_error:
            self._compute_error()
        if unaugment: #match the channels of the original matrix
            Xhat = Xhat[:self.Xraw.shape[0], :]
        return Xhat

    def _compute_error(self):
        """
        Computes the normalized error between the original augmented matrix X
        and the reconstructed matrix Xhat using the Frobenius norm.

        Returns
        --------
        E : float
            The frobenius norm of the differences between the two
            matrices normalized by the frobenius norm of Xaug

        """
        #FIXME assert these can be properly subtracted from eachother
        #FIXME for jovanovich method
        o_shape = self.Xraw.shape #original shape
        matched_Xhat = self.Xhat[:o_shape[0], :o_shape[1]]
        #normalized error via frobenius
        self.E = LA.norm(self.Xraw - matched_Xhat,
                'fro') / LA.norm(self.Xraw, 'fro')
        return self.E

    def spectrum(self, sort=False, sort_modes=False, plotfig=False,
            savefig=False, freq_space=None):
        """
        Compute the DMD spectrum from outputs of DMD.fit

        Parameters
        ----------
        sort : string
            Sort type, either by 'frequencies' or by 'power'
            (lowest to highest). Sort the other via the reordering of
            the specified.
        sort_modes: boolean
            Rearrange Phi modes to match any sorting of
            frequencies or power
        plotfig : boolean
            Plot a spectrum of f and P
        savefig : boolean
            Save the plotted spectrum of f and P
        freq_space : array
            Discretize f according to freq_space; otherwise keep
            original frequencies non discretized

        Returns
        -------
        f : array-like
            The (sorted) frequencies of the modes in cycles/sec
        P : array-like
            The (sorted) power of the modes

        """
        #check input
        if self.mu is None:
            warnings.warn("`DMD.fit` must be run before `DMD.spectrum`")
        if sort: assert(sort in ['frequencies', 'power'])
        if sort_modes: assert(sort)

        f = np.abs(np.imag(self.mu) / (2 * np.pi))
        if freq_space is not None:
            discretized_f = list()
            for raw_f in f:
                idx = self._find_nearest_idx(freq_space, raw_f)
                discretized_f.append(freq_space[idx])
            f = discretized_f

        # roughly scales like the fft spectrum power
        P = np.array([LA.norm(self.Phi[:,i],2) ** 2 for i in
            range(self.Phi.shape[1])])

        #below theoretic. equivalent but produces wrong answer:
        #P = abs(np.diag(self.Phi.T.dot(self.Phi)))
        #use jovanovich amplitudes
        if self.jovanovich or self.condensed_jovanovich:
            P = self.alphas

        #sort P and modes 
        if sort:
            f, P, indices = self._sort(f, P, sort)
        if sort_modes:
            self.Phi = self.Phi[:, indices]
            self.mu = self.mu[indices]
            self.lambdas = self.lambdas[indices]
        if plotfig:
            plt.figure()
            plt.stem(f, P, 'k')
            title = 'DMD spectrum'
            plt.title(title)
            plt.xlabel('Frequency')
            plt.ylabel('DMD scaling')
            plt.show()
            if savefig:
                plt.savefig('%s.png' % title.replace(' ', ''))
        return f, P

    def _find_nearest_idx(self, array, value):
        """ Return the index of array that is closest to value.

        Parameters
        --------
        array : Array-like
            List 
        value : float
            Value

        Returns
        --------
        idx : float
            Index into 'array'
        """
        return (np.abs(np.array(array) - value)).argmin()

    def _sort(self, f, P, sort):
        """Sort frequencies and power according to one or the other
        according to `sort`

        Parameters
        --------
        f : Array-like
            List of frequencies
        P : Array-like
            List of powers
        sort : String
            Indicate to sort both frequency and power in
            nondecreasing order either by the frequencies given
            or by the power given.  Valid options: 'frequencies', 'power' 

        Returns
        --------
        f : Array-like
            Sorted frequencies
        P : Array-like
            Sorted powers
        """
        if sort not in ['frequencies', 'power']: raise ValueError
        assert(len(f) == len(P))
        if sort is 'frequencies':
            indices = list(range(len(f)))
            indices.sort(key=f.__getitem__)
        elif sort is 'power':
            indices = list(range(len(f)))
            indices.sort(key=P.__getitem__)
        f = [f[i] for i in indices]
        P = [P[i] for i in indices]
        return f, P, indices

    def _estimate_stack_factor(self, features, timesteps):
        """
        Back of the envelope estimation for choosing stack 
        number for minimizing reconstruction error*.  Choose
        the stack number such that the features/rows of Xaug will
        be at least twice the columns/timesteps of Xaug
        if possible.

        Parameters
        --------
        features : int
            The number of features of Xraw
        timesteps : int
            The number of timesteps of Xraw

        Returns
        --------
        stack_factor : int
            Number of times to stack Xraw row-wise

        Notes 
        -----
        Based on "Extracting Spatial-Temporal Coherent Patterns
        in Large-Scale Neural Recordings Using Dynamic Mode 
        Decompositions", Brunton et al. 2015.

        """
        self.stack_factor =  math.ceil(2 * float(timesteps) / features)
        #try stack factor of at least 2 for capturing freq content
        if self.stack_factor < 2: self.stack_factor = 2
        #ensure stack_factor does not eclipse...
        #original vector due to shifting
        if (timesteps - (self.stack_factor - 1)) < 2:
            #if stack would be less than two cols, set to half of timesteps
            self.stack_factor = int(timesteps / 2)
        #ensure stack_factor at least 1; no freq content captured at 1
        if self.stack_factor < 1: self.stack_factor = 1
        self.stack_factor = int(self.stack_factor)
        return self.stack_factor

    def plot_complex_spec(self, savefig=False):
        """
        Plot the complex spectrum.

        Parameters
        ----------
        savefig : boolean
            Save figure by title

        """
        title = 'lambdas'
        lim = 1.5
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.Circle((0, 0), radius=1, edgecolor='k', facecolor='None')
        plt.scatter(self.lambdas.real, self.lambdas.imag, 'rk')
        plt.xlim(-lim, lim)
        plt.ylim(-lim, lim)
        plt.xlabel(r'$\mathbb{C}$')
        plt.ylable(r'')
        plt.show()
        if savefig:
            plt.savefig('%s.png' % title.replace(' ', ''))

    def _estimate_optimal_SVHT(self, raw_shape, diagS):
        """
        Estimate the optimal hard threshold, sets `self.r` 
        accordingly.

        Parameters
        --------
        raw_shape : tuple
            The shape of Xaug or Xraw
        diagS : array-like
            Diagonal values of S

        Notes
        -----
        Adapted from Matlab: "Optimal Hard Threshold for Singular
        Values is 4 / sqrt(3)". Gavish and Donoho

        """
        if len(diagS.shape) != 1:
            raise ValueError("`diagS` must be array-like")
        #get the ratio of features / timesteps
        beta = raw_shape[0] / raw_shape[1]
        #if features > timesteps then use timesteps / features instead
        if beta > 1: beta = 1 / beta
        #sigma flag unknown (0)
        omega = self._optimal_SVHT_coef(beta, 0) * np.median(diagS)
        self.r = sum(diagS > omega) #update r
        #adjust estimate to at least 2 allows complex components
        if self.r < 2: self.r = 2 

    def _incMarPas(self, x0, beta, gamma):
        """ 

        Notes
        -----
        Adapted from Matlab: "Optimal Hard Threshold for Singular
        Values is 4 / sqrt(3)". Gavish and Donoho

        """
        assert(beta <= 1)
        topSpec = (1 + np.sqrt(beta)) ** 2
        botSpec = (1 - np.sqrt(beta)) ** 2
        _MarPas = lambda x: self._ifElse((topSpec - x) * (x - botSpec) > 0,
                np.sqrt((topSpec - x) * (x - botSpec)) / (beta * x) / (2 * np.pi),
                0)
        if gamma:
           fun = lambda x: (x ** gamma * _MarPas(x))
        else:
           fun = lambda x: _MarPas(x)
        #note this does not implement Lobatto quadrature
        I, err = sio.quad(fun, x0, topSpec)
        return I

    def _ifElse(self, Q, point, counterPoint):
        """ 

        Notes
        -----
        Adapted from Matlab: "Optimal Hard Threshold for Singular
        Values is 4 / sqrt(3)". Gavish and Donoho

        """
        y = point
        if not Q:
            y = counterPoint
        return y
        #indices = map(operator.not_, Q)
        #if any(indices):
            #if len(counterPoint) == 1:
                #counterPoint = np.ones((Q.shape)) * counterPoint
            ##FIXME
            #y[indices] = counterPoint[indices]

    def _MedianMarcenkoPastur(self, beta):
        """ 

        Notes
        -----
        Adapted from Matlab: "Optimal Hard Threshold for Singular
        Values is 4 / sqrt(3)". Gavish and Donoho

        """
        _MarPas = lambda x: 1 - self._incMarPas(x, beta, 0)
        lobnd = (1 - np.sqrt(beta)) ** 2.0
        hibnd = (1 + np.sqrt(beta)) ** 2
        change = 1
        while change and (hibnd - lobnd > .001):
          change = 0
          x = np.linspace(lobnd,hibnd,5)
          y = np.zeros((x.shape))
          for i, xi in enumerate(x):
              y[i] = _MarPas(xi)
          if any(y < 0.5):
             lobnd = np.max(x[y < 0.5])
             change = 1
          if any(y > 0.5):
             hibnd = np.min(x[y > 0.5])
             change = 1
        med = (hibnd + lobnd) / 2
        return med

    def _optimal_SVHT_coef_sigma_known(self, beta):
        """ 

        Notes
        -----
        Adapted from Matlab: "Optimal Hard Threshold for Singular
        Values is 4 / sqrt(3)". Gavish and Donoho

        """
        assert(beta>0)
        assert(beta<=1)
        #assert(all(beta>0))
        #assert(all(beta<=1))
        #assert(type(beta) == np.ndarray) # beta must be a vector
        w = (8 * beta) / (beta + 1 + np.sqrt(beta ** 2 + 14 * beta +1)) 
        lambda_star = np.sqrt(2 * (beta + 1) + w)
        #assert(lambda_star.shape == beta.shape)
        return lambda_star

    def _optimal_SVHT_coef_sigma_unknown(self, beta):
        """ 

        Notes
        -----
        Adapted from Matlab: "Optimal Hard Threshold for Singular
        Values is 4 / sqrt(3)". Gavish and Donoho

        """
        assert(beta>0)
        assert(beta<=1)
        #assert(all(beta>0))
        #assert(all(beta<=1))
        #assert(type(beta) == np.ndarray) # beta must be a vector
        #get lambda star
        coef = self._optimal_SVHT_coef_sigma_known(beta)
        #MPmedian = np.zeros((beta.shape))
        #for i, bi in enumerate(beta.shape):
            #MPmedian[i] = self._MedianMarcenkoPastur(bi)
        MPmedian = self._MedianMarcenkoPastur(beta)
        omega = coef / np.sqrt(MPmedian)
        #assert(omega.shape == beta.shape)
        return omega

    def _optimal_SVHT_coef(self, beta, sigma_known):
        """ Computes the optimal threshold r elements to cut off
        in the Singular Value Decompositions (SVD). Updates self.r
        to reflect this new value.

        Parameters
        ----------
        beta : scalar or array-like
           Aspect ratio m/n of the matrix to be denoised, 0<beta<=1. 
        sigma_known : boolean
           Flag: 1 if noise level known, 0 if unknown

        Returns
        -------
        r : float
            The cutoff element

        Notes
        -----
        Adapted from Matlab: "Optimal Hard Threshold for Singular
        Values is 4 / sqrt(3)". Gavish and Donoho

        """
        if sigma_known:
            coef = self._optimal_SVHT_coef_sigma_known(beta)
        else:
            coef = self._optimal_SVHT_coef_sigma_unknown(beta)
        return coef
