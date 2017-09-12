#Author : Karl Marrett kdmarrett@gmail.com

from sklearn import preprocessing
import numpy as np
from DMD import DMD
from matplotlib import pylab as plt
import numpy.linalg as LA
import warnings

class mrDMD:

    def __init__(self, kwargs, method='frequency_bin',
            subsample=False, L=10, cutoff_mode_thresh=None,
            freq_space='default', freq_gradation='default', 
            plot_levels=False, subtraction=False,
            excess_reconstruction=False, power_denoise=False,
            power_denoise_thresh=.05, lower_lim=0.0,
            linear_freq_bins=False, l2_norm=False):
        """ 
        Multi-Resolution Dynamic Mode Decomposition (mrDMD).

        Applies DMD in windowed fashion similar in implementation to Short-Time
        FFT.  Yields time-evolving spatial modes with corresponding
        growth and frequency characteristics.

        Parameters
        --------
        kwargs : dict
            The dictionary of keyword arguments for the 
            dmd instance to be run at each MRDMD window.
            Note: 'dt' of MRDMD is determined by the value 'kwargs['dt']'
            since these values must match between DMD and MRDMD.
        method : string
            Determines the methodology MRDMD via passing
            one of the following values:
            'slow_cutoff' : 
            'frequency_bin' :  
            'power_cutoff : 
        subsample : boolean
            Each level of MRDMD where DMD is applied 
            is only sensitive to a subset of the frequency range.
            The higher limit of a level's frequency range is often
            much lower than the Nyquist frequency of the input raw data.
            When subsample is set to True and the upper limit of a given
            level is below Nyquist of the data's sampling frequency, the data 
            is subsampled where possible.
        cutoff_mode_thresh : int
            Manually set the number of modes to take for each level.  This
            only applies to the methods of 'slow_cutoff' and 'power_cutoff'
        freq_space : array-like
            The list of frequencies to discretize modes into. The 
            continuos frequency computed is rounded to nearest value
            in this list.
        freq_gradation : int
            When freq_space is 'default', 'freq_gradation' determines
            the number of frequencies to split the frequency space into
            between 0 and the maximum frequency.
        plot_levels : boolean
            Print a plot of frequencies and modes recorded at each window.
        subtracton : boolean
            Each level records a set of modes.  When 'subtraction' is True,
            the reconstruction of the time window from the recorded modes
            is subtracted from the raw data over the time window such that
            higher levels ideally do not record the same modes again. In
            practice, this strategy tends to amplify noise at the higher 
            levels.
        excess_reconstruction : boolean
            As an alternative to subtracting the recorded modes which 
            may amplify noisy or erroneous modes via subtraction on the data,
            this strategy makes a reconstruction of the remaining
            non-recorded modes for higher level time windows.
        power_denoise : boolean
            'power_denoise' seeks to filter out noise via the estimation of
            power associated with each mode. If 'power_denoise' is True, 
            only modes with power representing a percentage of the 
            total power above 'power_denoise_thresh' will be recorded.
        power_denoise_thresh : float
            The threshold percentage of total power to keep modes when
            'power_denoise' is set. See 'power_denoise'.
        lower_lim : float
            The lower frequency of the first level.
        linear_freq_bins : boolean
            In the 'frequency_bin' method, each level only records modes with
            associated frequencies within a designated frequency range.  When
            'linear_freq_bins' is True, this frequency range is divided
            linearly between each level. Otherwise it is divided
            logarithmically, since the length of the time window(s) of each
            level is half as long as the preceding window.

        Returns
        --------
        mrDMD : object
            The blank instance of mrDMD with given parameters.

        See Also
        --------
        class: `DMD` 

        """
        #user input handling
        assert(method is 'slow_cutoff' or method is 'frequency_bin'
                or method is 'power_cutoff')
        if method is 'frequency_bin':
            if cutoff_mode_thresh is not None:
                warnings.warn('cutoff_mode_thresh does not apply with' +\
                        'frequency_bin method')
        else:
            if cutoff_mode_thresh is None:
                #when X is 1 channel stack_factor determines the number
                #of modes to extract at each level. Each mode is a
                #pair (complex, real), the default behavior below
                #thus takes a cutoff at half of the total modes
                #in single channel examples.
                if ('stack_factor' in kwargs) and \
                    (kwargs['stack_factor'] == int):
                    cutoff_mode_thresh = int((kwargs['stack_factor'] / 2) / 2)
            else:
                if not (0.0 < cutoff_mode_thresh <= 1.0):
                    raise('`cutoff_mode_thresh` must be 0.0 < cutoff_mode_thresh <= 1.0')
            if subsample:
                warnings.warn('subsample does not apply with method %s' % method)
                subsample = False

        if ('stack_factor' in kwargs) and \
            (kwargs['stack_factor'] == int) and \
            (kwargs['stack_factor'] < 2):
            warnings.warn('stack_factor must always be at least 2 or greater to capture frequency content')

        if ('stack_factor' in kwargs) and \
                (kwargs['stack_factor'] == int):
            self.stack_factor = kwargs['stack_factor']
        else:
            self.stack_factor = 'estimate'

        if not ('dt' in kwargs):
            warnings.warn('dt unspecified in kwargs. Using default dt = 1 second')

        #highest frequency to record from 
        self.max_freq = 200
        self.freq_space = freq_space
        self.linear_freq_bins = linear_freq_bins
        self.freq_gradation = freq_gradation
        self.cutoff_mode_thresh = cutoff_mode_thresh
        self.dmd = DMD(**kwargs)
        self.dt = self.dmd.dt
        self.subsample = subsample
        self.L = L
        self.excess_reconstruction = excess_reconstruction
        self.method = method
        self.spec = None
        #The list of the lower and upper frequency bounds of each 'L' levels
        self.freq_bins = np.zeros((L, 2))
        #the lower frequency bound of the first level
        #all recorded modes are therefore above this frequency
        self.lower_lim = lower_lim
        self.time_resolution = None
        self.time_steps = None
        self.levelSet = set()
        self.plot_levels = plot_levels
        self.subtraction = subtraction
        self.power_denoise = power_denoise
        self.power_denoise_thresh = power_denoise_thresh
        self.Phi_modes = None #freq by time by feature
        self.original_channels = None
        self.original_samples = None
        self.l2_norm = l2_norm
        self.baseline_power_denoise = None

    def _in_range(self, x, lim): 
        """ 
        Return the indices of the x vector where
        the values are within the range of lim.

        Parameters
        --------
        x : vector-like
            The list or vector
        lim : vector-like
            The list or vector of lower and upper limit respect.
            The lower limit is inclusive, the upper exclusive
            such that no frequency is overlapping in separate 
            mrDMD levels (`l`).

        Returns
        --------
        indices : vector-like
            The indices corresponding to such values of x.

        """
        return [i for i, v in enumerate(x) if (v >= \
            lim[0]) and (v < lim[1])]

    def _scalar_add(self, array, value): 
        """ 
        Element-wise addition to an array, list, iterable
        """
        return [x + value for x in array]

    def fit(self, X):
        """ 
        Public call to fit mrDMD to a data sample X.

        Parameters
        --------
        X : matrix-like
            The data matrix 

        Returns
        --------
        self.spec : matrix-like
            The spectrogram of shape : (len(self.freq_space), self.time_steps)

        See Also
        --------
        :class:`mrDMD._fit` : private call on `fit`

        """
        if self.method == 'frequency_bin':
            # Assign frequency bins for each level
            if self.linear_freq_bins: #deprecated
                if self.freq_gradation is 'default':
                    self.freq_space = np.linspace(0, self.max_freq, self.max_freq + 1)
                else:
                    self.freq_space = np.linspace(0, self.max_freq, self.freq_gradation)
                self.freq_bin_width = len(self.freq_space) // self.L
                for l in range(self.L):
                    #get the indices of freq_space limits
                    lim_inds = [l * self.freq_bin_width, (l + 1) * self.freq_bin_width]
                    if l == (self.L - 1): # include all frequencies
                        lim_inds[1] = len(self.freq_space) - 1
                    #get the values of the lower and upper discretized freqs
                    lim = (self.freq_space[lim_inds[0]], self.freq_space[lim_inds[1]])
                    self.freq_bins[l, :] = lim # save freq lim information
            else:
                #assign frequency bins based off time of window
                for l in range(self.L):
                    if l == 0:
                        time_of_window = X.shape[1] * self.dt
                    else:
                        time_of_window /= 2
                    lower_lim = (2 * float(time_of_window)) ** -1
                    upper_lim = (float(time_of_window)) ** -1
                    #get the values of the lower and upper discretized freqs
                    lim = (lower_lim, upper_lim)
                    self.freq_bins[l, :] = lim # save freq lim information

        # Assign discretized frequency space
        if self.freq_space is 'default':
            if self.freq_gradation is 'default':
                self.freq_space = np.linspace(0, self.max_freq, self.max_freq + 1)
            else:
                #self.freq_space = np.linspace(0, np.max(self.freq_bins), (((2 ** L) + 1) // freq_gradation))
                self.freq_space = np.linspace(0, self.max_freq, self.freq_gradation)

        sample_resolution = X.shape[1] * (2 ** (-self.L))
        self.time_steps = int(np.ceil(X.shape[1] / sample_resolution))
        self.time_resolution = sample_resolution * self.dt
        self.levelSet = set()
        self.original_channels = X.shape[0]
        self.original_samples = X.shape[1]
        self.spec = np.zeros((len(self.freq_space), self.time_steps))
        #freq by time by feature
        self.Phi_modes = np.zeros((len(self.freq_space),
            X.shape[0], self.time_steps))
        time_idx = range(self.time_steps)
        l = 0
        half = 'left'
        self.spec = self._fit(X, l, time_idx, half)

        if self.method is 'frequency_bin':
            self.summed_Phi_modes = np.zeros((self.L,
                self.Phi_modes.shape[1], self.Phi_modes.shape[2]))

        return self.spec

    def _fit(self, X, l, time_idx, half, nyquist_factor=12):
        """ 
        Private recursive method for fitting mrDMD to a data sample.

        Parameters
        --------
        X : matrix-like
            The data matrix
        l : scalar
            The local level of the mrDMD. l is in [0, L)
        time_idx : list
            The indices detailing the position of the windowed X relative
            to the initial full X matrix passed to mrDMD.fit()

        Returns
        --------
        self.spec : matrix-like
            The spectrogram

        See Also
        --------
        :class:`mrDMD.fit` : public call on `fit`

        """
        if l < self.L:
            #make sure all changes to dt are contained via hard reset
            self.dmd.dt = self.dt
            #record the original samples before subsampling
            local_t_shots = X.shape[1] 
            if self.subsample: #subsample compress each window
                upper_lim = self.freq_bins[l, 1]
                upper_dt = 1.0 / (nyquist_factor * upper_lim) #dictated by Nyquist
                stride = int(upper_dt / self.dt)
                if stride < 1:
                    #leave dt, stride at its non subsampled value
                    stride = 1
                    warnings.warn('Levels %d and higher are not subsampled' % l)
                self.dmd.dt *= stride #adjust dt according to M
                Xsub = X[:, ::stride].copy()
            else:
                Xsub = X

            try:
                if not Xsub.any():
                    warnings.warn('X is a zero vector. mrDMD stopped' +\
                        'with X shape %d by %d at level %d\n' % (X.shape[0], X.shape[1], l))
                    return self.spec #catch and add nothing
                if Xsub.size == 0:
                    warnings.warn('X is an empty vector. mrDMD stopped' +\
                        'with X shape %d by %d at level %d\n' % (X.shape[0], X.shape[1], l))
                    return self.spec #catch and add nothing
                if self.stack_factor == 'estimate':
                    local_stack_factor = self.dmd._estimate_stack_factor(*Xsub.shape)
                else:
                    local_stack_factor = self.stack_factor
                if (Xsub.shape[1] - local_stack_factor - 1) < 2:
                    warnings.warn('X is too small of a vector. mrDMD stopped' +\
                        'with X shape %d by %d at level %d\n' % (X.shape[0], X.shape[1], l))
                    return self.spec #catch and add nothing
                self.dmd.stack_factor = self.stack_factor #must explicitly reset stack_factor for each fit
                self.dmd.fit(Xsub)
                if self.method is 'slow_cutoff' or self.method is 'frequency_bin':
                    #any sorting of f, P also sorts the corresponding self.dmd.Phi
                    f, P = self.dmd.spectrum(freq_space=self.freq_space,
                            sort='frequencies', sort_modes=True)
                elif self.method is 'power_cutoff':
                    #any sorting of f, P also sorts the corresponding self.dmd.Phi
                    f, P = self.dmd.spectrum(freq_space=self.freq_space,
                            sort='power', sort_modes=True)
            except LA.LinAlgError as err:
                ##dmd will fail on eig of Ahat when passed a zero vector
                if err.message == "SVD did not converge":
                    warnings.warn("DMD failed with X shape %d by %d at level %d\n %s\n"
                            % (X.shape[0], X.shape[1], l,
                                err.message))
                else:
                    warnings.warn("DMD failed with X shape %d by %d at level %d\n X may be a zero vector\n %s\n"
                            % (X.shape[0], X.shape[1], l,
                                err.message))
                return self.spec #catch and add nothing
            #except ValueError as err:
                ##dmd will fail on svd when data matrices are too large
                ##caused by some unknown failure in lapack
                #warnings.warn("DMD failed with X shape %d by %d at level %d\n X may be a zero vector"
                        #% (X.shape[0], X.shape[1], l))
                ##import pdb; pdb.set_trace()
                #return self.spec #catch and add nothing
            except AssertionError:
                #dmd will fail when the vector length < stack_factor
                warnings.warn("DMD failed with X shape %d by %d at level %d.\n Vector length may be < stack_factor"
                        % (X.shape[0], X.shape[1], l)
                        + "\nConsider lowering L or dmd.stack_factor")
                return self.spec #catch and add nothing
            #except ValueError as err:
                ##print(err.args)
                #warnings.warn("Lambdas contain no complex components")
                #return self.spec #catch and add nothing
            #except:
                ##dmd will fail when the vector length < stack_factor
                #warnings.warn("DMD failed with X shape %d by %d at level %d.\n CVXPY SolverError"
                        #% (X.shape[0], X.shape[1], l)
                        #+ "\nConsider lowering L or dmd.stack_factor")
                #return self.spec #catch and add nothing
            X_next_l = self._reconstruct(X, l, half, time_idx, local_t_shots, f, P)
            split_ind_time = local_t_shots // 2
            split_ind = len(time_idx) // 2
            first_idx = range(split_ind)
            second_idx = range(split_ind, len(time_idx))
            #import pdb; pdb.set_trace()
            first_idx = self._scalar_add(first_idx, time_idx[0])
            second_idx = self._scalar_add(second_idx, time_idx[0])
            #import pdb; pdb.set_trace()
            self.spec = self._fit(X_next_l[:, :split_ind_time], l+1,
                    first_idx, half='left')
            self.spec = self._fit(X_next_l[:, split_ind_time:], l+1, second_idx,
                    half='right')
        return self.spec

    def _record_power_and_phi(self, fP, time_idx, Phi, selected_mode_idx):
        """ 
        Summate spectral power content to self.spec and record Phi 
        weightings (the spatial modes) at the given time interval.
        
        Parameters
        --------
        fP : array-like 
            The arrary or tuple of frequency followed by its
            respective power to be recorded
        time_idx : list
            The indices detailing the position of the windowed X relative
            to the initial full X matrix passed to mrDMD.fit()
        Phi : matrix-like
            Spatial weightings
        selected_mode_idx : list
            Contains list of indices into the modes of Phi to
            include

        """
        for i in selected_mode_idx:
            fP_tup = fP[i]
            idx = self.dmd._find_nearest_idx(self.freq_space, fP_tup[0])
            self.spec[idx, time_idx] += fP_tup[1]
            #record Phi
            if self.l2_norm:
                self.Phi_modes[idx, :, time_idx] += preprocessing.normalize(
                    np.abs(Phi[:self.original_channels, i].reshape(1, -1)), 
                    norm='l2').flatten()
            else:
                self.Phi_modes[idx, :, time_idx] += \
                    np.abs(Phi[:self.original_channels, i].reshape(1, -1))

    def _reconstruct(self, X, l, half, time_idx, local_t_shots,
            f, P, savefig=False, verbose=True):
        """ 
        Reconstruct X based on method of 'slow_cutoff' or
        'frequency_bin'.  'slow_cutoff' method takes the first 
        number of modes specified by the `slow_cutoff` parameter
        passed during initialization of mrDMD.  In the 'frequency_bin' method,
        each level of mrDMD will only extract frequencies
        within a certain range according to level and it's frequency bounds
        specified in 'self.freq_bins'.  The range is linearly determined by the
        window level of mrDMD.  

        The frequencies from the subtracted modes are recorded
        (added) to the spectogram.
        
        Parameters
        ----------
        X : matrix-like
            The data matrix
        l : scalar
            The local level of the mrDMD. l is in [0, L)
        time_idx : list
            The indices detailing the position of the windowed X relative
            to the initial full X matrix passed to mrDMD.fit()
        local_t_shots : scalar
            Number of (before subsampled) timeshots of the current data window
        f : vector-like
            Frequencies
        P : vector-like
            Power at each frequency
        savefig : boolean
            Saves figure in current directory with title given
        verbose : boolean
            Prints descriptive reconstruction behavior

        Returns
        --------
        X_next_l : matrix-like
            The new X matrix reconstructed from the remaining higher
            frequency modes.  This is the matrix to be passed onto
            the lower (shorter time period) windows.

        """
        #total number of modes
        total_modes = self.dmd.Phi.shape[1] 
        if self.cutoff_mode_thresh: 
            cutoff_mode = int(total_modes * self.cutoff_mode_thresh)
            if cutoff_mode > total_modes:
                warnings.warn('Cutoff mode %d exceeds number of modes' %\
                        cutoff_mode)
                cutoff_mode = total_modes
        else:
            cutoff_mode = total_modes

        assert(len(f) == total_modes)
        fP = np.array(list(zip(f, P)))

        if 'slow_cutoff' == self.method: #with any cutoff method
            selected_mode_idx = range(cutoff_mode)
            excess_mode_idx = range(cutoff_mode, total_modes)
            assert(len(selected_mode_idx) == cutoff_mode)
        elif self.method == 'power_cutoff':
            #modes sorted in nondecreasing order according to power 
            first_mode = total_modes - cutoff_mode
            #excess_mode_idx = range(first_mode)
            #selected_mode_idx = range(first_mode, total_modes)
            selected_mode_idx = range(cutoff_mode)
            excess_mode_idx = range(cutoff_mode, total_modes)
            assert(len(selected_mode_idx) == cutoff_mode)
        elif self.method == 'frequency_bin':
            lim = self.freq_bins[l, :].copy() #get the freq lim information
            selected_mode_idx = self._in_range(f, lim)

        #Denoise: do not reconstruct or record modes
        #with power below threshold
        if self.power_denoise:
            tot_energy = sum(P)
            keep_ind = [ind for ind, p in enumerate(P) if (ind in
                selected_mode_idx) and ((p / tot_energy) > self.power_denoise_thresh)]
            selected_mode_idx = keep_ind
            #reupdate the high modes

        #only record modes with powers above some baseline
        if self.baseline_power_denoise is not None:
            #take only modes with higher P than average rest, at corr. f
            selected_mode_idx = [i for i in selected_mode_idx
                    if P[i] > self.baseline_power_denoise[self.dmd._find_nearest_idx(self.freq_space, 
                        f[i])]]
        select_mode_set = set(selected_mode_idx)
        excess_mode_idx = [i for i in range(total_modes) if i not in select_mode_set]

        if selected_mode_idx == []: #if no modes selected, return
            if self.plot_levels:
                print("No power recorded with X shape %d by %d at level %d"
                        % (X.shape[0], X.shape[1], l))
            return X #subtract nothing add no power

        self.dmd.dt = self.dt # hard reset for reconstruction
        if self.subtraction:
            #make a reconstruction for the time window
            #using on the selected low modes or modes within
            #the frequency bounds
            Xhatlow = self.dmd.transform(timesteps=local_t_shots,
                    keep_modes=selected_mode_idx)
            Xhathigh = None
            assert(X.shape == Xhatlow.shape)
            #subtract this reconstruction from the window
            #to create the data for lower levels
            X_next_l = X - Xhatlow
        else:
            X_next_l = X 
            Xhathigh = None
            Xhatlow = None
        if self.excess_reconstruction:
            #make a reconstruction based off of all 
            #unselected modes used
            #to create the data window for all lower levels
            Xhathigh = self.dmd.transform(local_t_shots,
                    keep_modes=excess_mode_idx)
            Xhatlow = None
            X_next_l = Xhathigh

        #FIXME
        #selected_mode_idx = selected_mode_idx[::2] #record only one of the real imag pair
        fP_record = fP[selected_mode_idx]
        #record only the power of the modes being subtracted
        self._record_power_and_phi(fP, time_idx, self.dmd.Phi, selected_mode_idx)
        #show plots for levels)
        if self.plot_levels and (l not in self.levelSet) and (half == 'left'):
            self._draw_levels(f, P, l, savefig, time_idx,
                Xhatlow, Xhathigh, fP_record, X, X_next_l, verbose)
        return X_next_l

    def _scale1(self, arr, mval=None):
        """ 
        Scale a vector by its maximum value or by some value
        specified by mval.
        
        Parameters
        --------
        arr : array-like
            The data matrix
        mval : scalar (Optional)
            The scalar value to divide by.  Default is None which
            searches the array for the maximum value to divide
            by.

        Returns
        --------
        arr : array-like
            The new scaled array.

        """
        if mval is None:
            mval = np.max(arr)
        return np.array([i / mval  for i in arr])

    def _day_plot(self, time_idx, x, title='', step=1, color='k',
            savefig=False):
        """ 
        Draw a day plot (lines for each row of x separated by distance `step`) 
        representation of matrix `x`

        Parameters
        ----------
        time_idx : array-like
            array of time indices for the current window
        x : matrix
            The data to display (typically timeseries data)
        title : string
            Title to be printed to plot. Also name of saved file
        step : float
            Distance on y-axis to separate each time varying channel
        color : string
            Color of lines
        savefig : boolean
            Saves figure to pdf

        """
        plt.figure()
        plt.rc('text', usetex=True)
        for ri, row in enumerate(x):
            current_step = ri * step
            y = x[ri, :]
            scaled = self._scale1(y)
            shifted = scaled + current_step
            if (x.shape[1] == len(time_idx)):
                plt.plot(time_idx, shifted, color)
            else:
                plt.plot(shifted, color)
        plt.ylim([-1, x.shape[0] * step])
        plt.ylabel('Channels')
        plt.xlabel('Time (s)')
        plt.title(title)
        plt.show()
        if savefig:
            plt.savefig('%s.pdf' % title)
        plt.close()

    def _draw_levels(self, f, P, l, savefig, time_idx,
        Xhatlow, Xhathigh, fP_record, X, X_next_l, verbose):
        """ 
        Draw descriptive plots at each level of mrDMD showing
        the spatial modes and spectral components recorded
        and/or extracted.

        Parameters
        ----------
        f : array-like
            Frequencies
        P : array-like
            Corresponding power
        l : int
            The level of mrDMD currently on
        savefig : boolean
            Saves figure
        time_idx : array_like
            The array of indices associated with current window
        Xhatlow : matrix
            The matrix representation of the unselected matrices
        Xhathigh : matrix
            The matrix representation of the selected matrices
        fP_record : list
            The list of (f, P) tuple pairs recorded
        X : matrix
            The raw data matrix
        X_next_l :
            The data matrix to be passed to lower levels
        verbose : boolean
            Explicit printing of f and P recorded

        """
        self.levelSet.add(l) #only plot once per level
        if verbose:
            print('\n\nLevel: %d' % l)
        #Frequency vs. scalings of DMD
        plt.figure()
        plt.rc('text', usetex=True)
        plt.stem(f, P, 'k')
        plt.title(r'Level: %d' % l)
        title = r'SpectrumLevel:%d' % l
        plt.xlabel(r'Frequency')
        plt.ylabel(r'DMD scaling')
        plt.show()
        if savefig:
            plt.savefig('%s.pdf' % title)
        plt.close()
        if verbose:
            print('Power recorded [frequency, Power]:')
            print(fP_record)
            print('Added to time_idx %d - %d' % (time_idx[0],
                time_idx[-1]))

        #show the original time window of X in black, then
        #show the subtracted reconstruction or the high mode
        #reconstruction in red
        if self.original_channels == 1:
            plt.figure()
            plt.rc('text', usetex=True)
            plt.plot(time_idx, X[0,:], 'k', label=r'\left| X \right|')
            if self.subtraction:
                plt.plot(time_idx, Xhatlow[0,:], 'r',
                        label=r'\left| \hat{X} \right|')
            else:
                plt.plot(time_idx, Xhathigh[0,:], 'r',
                        label=r'\left| \hat{X} \right| left over modes')
            plt.legend()
            title = r'Reconstruction level: %d' % l
            plt.title(title)
            plt.show()
        else:
            self._day_plot(time_idx, X, title=r'\left| X \right|, level: %d' % l,
                    savefig=savefig)
            if self.subtraction:
                title_end=r'\left| \hat{X} \right|, level: %d' % l
                self._day_plot(time_idx, Xhatlow, color='r', title=title_end,
                        savefig=savefig)
            elif self.excess_reconstruction:
                title_end=r'\left| \hat{X} \right| left over modes, level: %d' % l
                self._day_plot(time_idx, Xhathigh,
                        color='r', title=title_end,
                        savefig=savefig)

        #show the reconstruction for the next levels only for
        #subtraction constructed via: x - xhat
        if self.subtraction:
            if self.original_channels == 1:
                plt.figure()
                plt.rc('text', usetex=True)
                plt.plot(time_idx, X_next_l[0,:], 'b', label=r'\left| X \right|')
                title = r'\left| X - \hat{X} \right| level: %d' % l
                plt.title(title)
                plt.show()
                if savefig:
                    plt.savefig('%s.pdf' % title)
                plt.close()
            else:
                title = r'\left| X - \hat{X} \right| level: %d' % l
                self._day_plot(time_idx, X_next_l, color='b',
                        title=title, savefig=savefig)

        #show heatmap so far
        self.heatmap(title='Level %d' % l)

    def heatmap(self, title='', norm=False, savefig=False,
            xticks_loc='default', xticks_label='default',
            yticks_interval='default'):
            #xticks_loc=np.linspace(0,3000, 7),
            #xticks_label=np.arange(0, 35, 5), yticks_interval=5):
        """
        Draw a heatmap of the spectral information returned by
        mrDMD.fit()

        Parameters
        --------
        title : string
            Titles the plot and names the png file
        norm : boolean
            The local level of the mrDMD. l is in [0, L)
        savefig : boolean
            Saves figure in current directory with title given
        xticks_loc : array-like
            Locations of xticks (time axis)
        xticks_label : array-like
            Labels at xticks (time axis)
        yticks_interval : int
            Skip interval for yticks (frequency axis)

        """
        if xticks_loc is 'default':
            t = range(self.time_steps) * np.tile(self.time_resolution, self.time_steps)
        plt.figure()
        if norm:
            spec = self.spec / np.max(self.spec)
        else:
            spec = self.spec
        plt.pcolor(t, self.freq_space, spec, cmap='hot')
        #plt.yticks(np.arange(len(self.freq_space) -
            #1)[::yticks_interval], [int(i) for i in
                #self.freq_space[::yticks_interval]])
        #plt.xticks(xticks_loc, xticks_label)
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (s)')
        plt.title(title)
        #plt.colorbar()
        if savefig:
            plt.savefig('%s.png' % title.replace(' ', ''))

