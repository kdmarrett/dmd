#Author : Karl Marrett kdmarrett@gmail.com

import numpy as np
from helperFunctions import *
from DMD import DMD
import pytest

#TODO stack_factor can be adjusted according to freq?
#TODO check all changes to stack_factor it needs to be tested
    #stack_factor may stay constant depends on augment
#TODO separate all functions that don't need fit to have run

class TestDMD:

    @pytest.fixture(scope="module")
    def dmd(self, request):
        """ Setup dmd fit on sine wave before any method of this 
        class"""
        dt = .01
        kwargs = {'dt':dt, 'stack_factor':'estimate',
                'scale_modes':True, 'use_optimal_SVHT':True}
        dmd = DMD(**kwargs)
        yield dmd
        print("Tear down dmd instance")
        del dmd

    @pytest.fixture(scope="function")
    def dmd_fitted(self, request):
        """ Setup dmd fit on sine wave before any method of this 
        class"""
        N = 1000
        t = np.linspace(0, 10, N)
        freqs = [3.3]
        size = 1
        dt = .01
        x = buildX(freqs, t, size=size)
        kwargs = {'dt':dt, 'stack_factor':'estimate',
                'scale_modes':True, 'use_optimal_SVHT':True}
        dmd = DMD(**kwargs).fit(x)
        yield dmd
        print("Tear down dmd instance")
        del dmd

    def test_augment_x(self, dmd):
        features_timesteps = [(100, 50), (50, 100), (50, 50), (1000, 3),
                (1, 50), (1, 2)]
        for feature_timestep in features_timesteps:
            feature, timestep = feature_timestep
            dmd.stack_factor = 'estimate'
            Xraw = np.random.rand(*feature_timestep)
            Xaug = dmd._augment_x(Xraw)
            assert(Xaug.shape[0] == dmd.stack_factor * feature)
            assert(Xaug.shape[1] == timestep - (dmd.stack_factor - 1))
            #check for shifting
            if dmd.stack_factor == 1:
                assert Xaug.shape == Xraw.shape
            elif dmd.stack_factor > 1:
                for i in range(min(feature * dmd.stack_factor - 1, Xaug.shape[1] - 1, 10)): #check at most 10 elems
                    row = i
                    col = Xaug.shape[1] - (i + 1)
                    assert(Xaug[row][col] == Xaug[row + feature][col - 1])
        dmd.stack_factor = -1
        with pytest.raises(ValueError):
            dmd._augment_x(Xraw)
        dmd.stack_factor = 1.3
        with pytest.raises(ValueError):
            dmd._augment_x(Xraw)

    def test_truncate(self, dmd):
        dmd.use_optimal_SVHT = False
        r_vals = (1, 5, 10, 1000)
        size = 100
        for r_val in r_vals:
            dmd.r = r_val
            U = np.random.rand(size, size)
            S = np.random.rand(size, size)
            V = np.random.rand(size, size)
            diagS = np.random.rand(size)
            dmd.diagS = np.random.rand(size)
            U, S, V = dmd._truncate(None, U, S, V, diagS)
            assert(r_val == dmd.r) #no side effects
            assert(U.shape[1] == r_val or U.shape[1] == size)
            assert(S.shape == (r_val, r_val) or S.shape == (size, size))
            assert(V.shape[1] == r_val or V.shape[1] == size)
            assert(len(dmd.diagS.shape) == 1)
            assert(len(dmd.diagS) == r_val or len(dmd.diagS) == size)
            
    def test_estimate_optimal_SVHT(self, dmd):
        features_timesteps = [(100, 50), (50, 100), (50, 50), (1000, 3),
                (1, 50)]
        for feature_timestep in features_timesteps:
            diagS = np.random.rand(feature_timestep[1])
            dmd._estimate_optimal_SVHT(feature_timestep, diagS)
            assert(dmd.r >= 2)
        diagS = np.random.rand(5,5)
        with pytest.raises(ValueError):
            dmd._estimate_optimal_SVHT(feature_timestep, diagS)
            
    def test_find_nearest_idx(self, dmd):
        nums = np.arange(3)
        for i, val in enumerate(range(3)):
            assert(dmd._find_nearest_idx(nums, val) == i)

    def test_estimate_stack_factor(self, dmd):
        """ Test stack_factor estimation for various input"""
        features_timesteps = [(100, 50), (50, 100), (50, 50), (1000, 3),
                (1, 50), (1, 2)]
        for feature_timestep in features_timesteps:
            feature, timestep = feature_timestep
            stack_factor = dmd._estimate_stack_factor(*feature_timestep)
            assert(dmd.stack_factor != 'estimate')
            assert(type(stack_factor) == int)
            #ensure stack_factor does not eclipse...
            #original vector due to shifting
            assert(timestep - (stack_factor - 1) > 1)

    def test_sort(self, dmd):
        """ test the _sort function of spectrum of DMD"""

        #test sort by power
        f = list(range(10))
        P = f[::-1] #inverted
        f_sort, P_sort, ind = dmd._sort(f, P, 'power')
        assert f_sort == P
        assert [f_sort[i] for i in ind] == f
        assert [P_sort[i] for i in ind] == P

        #test sort by frequency
        f = list(range(10))
        P = f[::-1] #inverted
        f_sort, P_sort, ind = dmd._sort(f, P, 'frequencies')
        assert P_sort == P
        assert [f_sort[i] for i in ind] == f
        assert [P_sort[i] for i in ind] == P

        #test incorrect sort String arg
        with pytest.raises(ValueError):
            dmd._sort(f, P, 'invalid string')

    def test_spectrum(self, dmd_fitted):
        #test freq_space
        freq_space = np.arange(10)
        f, P = dmd_fitted.spectrum(freq_space=freq_space)
        assert(len(f) == len(P))
        for fval in f:
            assert(fval in freq_space)

        #test sort power
        Phi_mu_lambdas = (dmd_fitted.Phi, dmd_fitted.mu, dmd_fitted.lambdas)
        f, P = dmd_fitted.spectrum(sort='power', sort_modes=True)
        last = P[0]
        for pval in P:
            pval >= last
        Phi_mu_lambdas_final = (dmd_fitted.Phi, dmd_fitted.mu, dmd_fitted.lambdas)
        for match in zip(Phi_mu_lambdas, Phi_mu_lambdas_final):
            assert(match[0].shape == match[1].shape)

        #test sort frequencies
        f, P = dmd_fitted.spectrum(sort='frequencies')
        last = f[0]
        for fval in f:
            fval >= last
        Phi_mu_lambdas_final = (dmd_fitted.Phi, dmd_fitted.mu, dmd_fitted.lambdas)
        for match in zip(Phi_mu_lambdas, Phi_mu_lambdas_final):
            assert(match[0].shape == match[1].shape)

    def test_fit(self, dmd_fitted):
        assert(dmd_fitted.Phi is not None)
        assert(dmd_fitted.lambdas is not None)
        assert(dmd_fitted.Atilde is not None)
        assert(dmd_fitted.mu is not None)
        if dmd_fitted.jovanovich or dmd_fitted.condensed_jovanovich:
            assert(dmd_fitted.Vand is not None)
            assert(dmd_fitted.alphas is not None)

    def test_reconstructions(self, dmd):
        pass

    def test_fit_transform(self, dmd, dmd_fitted):
        with pytest.raises(ValueError):
            dmd.transform()

    def test_transform(self, dmd, dmd_fitted):
        with pytest.raises(ValueError):
            dmd.transform()
        Xhat, E = dmd_fitted.transform(compute_error=True)
        z0, mu = dmd_fitted.z0, dmd_fitted.mu
        assert(Xhat.shape == dmd_fitted.Xraw.shape)
        assert(not Xhat.imag.any())
        #E normed by Xraw in DMD._compute_error thus below ~= < %1
        assert(E < 1)

        less_timesteps = 3
        total_timesteps = Xhat.shape[1] - less_timesteps
        Xhat_timestepped = dmd_fitted.transform(timesteps=total_timesteps)
        #FIXME
        #assert(np.array_equal(z0, dmd_fitted.z0))
        assert(np.array_equal(mu, dmd_fitted.mu))
        assert(Xhat_timestepped.shape[1] == Xhat.shape[1] - less_timesteps)
        assert(np.allclose(Xhat[:, :total_timesteps], Xhat_timestepped))

    def test_compute_error(self, dmd_fitted):
        dmd_fitted.transform()
        E = dmd_fitted._compute_error()
        assert(type(E) == np.float64)

        dmd_fitted.Xhat = dmd_fitted.Xraw.copy()
        E = dmd_fitted._compute_error()
        assert(E == 0.0)

    def check_frequency_dynamics(self, dmd, kwargs, X, freqs,
            rtol, growth_decay_list):
        """Test growth and decay of modes
        given signal of known dynamics"""
        dmd = DMD(**kwargs)
        dmd.fit(X)
        f, P = dmd.spectrum(sort='power', sort_modes=True)
        last_expected_mode = 2 * len(freqs) 
        #examine only dominant modes
        lambdas = dmd.lambdas[-last_expected_mode:]
        P = P[-last_expected_mode:]
        f = f[-last_expected_mode:]
        for freq, growth_decay in zip(freqs, growth_decay_list):
            f_idx = dmd._find_nearest_idx(np.asarray(f), freq)
            lambda_i = lambdas[f_idx]
            if growth_decay is 'stable':
                assert(np.isclose(abs(lambda_i), 1.0))
            elif growth_decay is 'decay':
                assert(abs(lambda_i) < 1.0)
            elif growth_decay is 'growth':
                assert(abs(lambda_i) > 1.0)

    def check_spatial_accuracy(self, dmd, kwargs, X, freqs, rtol):
        dmd = DMD(**kwargs)
        dmd.fit(X)
        #sort: highest power first
        f, P = dmd.spectrum(sort='power', sort_modes=True, plotfig=False)
        # two modes for each actual signal
        last_expected_mode = 2 * len(freqs) 
        original_channels = X.shape[0]
        Phi = np.abs(dmd.Phi[:original_channels])
        #check majority power within expected modes
        assert(np.isclose(np.sum(Phi), np.sum(Phi[:, -last_expected_mode:]), rtol=.01))
        #assert each chan increasing power like input signal
        for i in range(last_expected_mode):
            last = 0
            for val in Phi[:, -(i + 1)]:
                #check values are nondecreasing for chans
                assert(val >= last)
                last = val

    def check_spectrum_elements(self, dmd, kwargs, X, freqs, rtol):
        dmd = DMD(**kwargs)
        dmd.fit(X)
        #sort: highest power first
        f, P = dmd.spectrum(sort='power', plotfig=False)
        # two modes for each actual signal
        last_expected_mode = 2 * len(freqs) 

        #check that most dominant signals 
        # were contained in original signal
        for f_ind in f[-last_expected_mode:]:
            freqs_idx = dmd._find_nearest_idx(np.asarray(freqs), f_ind)
            assert(np.isclose(f_ind, freqs[freqs_idx], rtol=rtol))

        #check each original signal contained in final
        for freq in freqs:
            f_idx = dmd._find_nearest_idx(np.asarray(f), freq)
            assert(np.isclose(freq, f[f_idx], rtol=rtol))

    def test_frequency_spectrum(self, dmd):
        stack_factor = 'estimate'
        noise_amp = 0
        # relative difference allowance see np.isclose Notes
        rtol = .05 
        dt = .01
        N = 1000
        t = np.linspace(0, 10, N)
        freqs = primes(50)[::4]
        X = buildX(freqs, t, noise_amp=noise_amp)

        # Spectral test
        kwargs = {'dt':dt, 'stack_factor':stack_factor, 
            'scale_modes':True}
        self.check_spectrum_elements(dmd, kwargs, X, freqs, rtol)

        # Spatial test
        chans = 10
        X_multi = np.repeat(X, chans, axis=0) 
        # build spatial mask
        mask = np.diag([(chans ** -1) * i for i in range(chans)])
        #apply spatial mask
        X_spatial = mask.dot(X_multi) #power increases across channels
        self.check_spatial_accuracy(dmd, kwargs, X_spatial, freqs, rtol)

        # Spectral dynamic test
        X = buildX([freqs[0]], t, noise_amp=noise_amp) #first is stable
        #keep track of modes as either 'growth', 'decay', or 'stable'
        growth_decay_list = ['growth'] * len(freqs)
        growth_decay_list[0] = 'stable'
        for i, freq in enumerate(freqs[1:]):
            X_stable = buildX([freq], t, noise_amp=noise_amp)
            # build dynamic scaled matrix
            scale_mask = [np.e ** (i / N ) for i in range(N)]
            if i % 2 == 1:
                scale_mask = scale_mask[::-1] #alternating growing/decaying modes
                growth_decay_list[i + 1] = 'decay'
            #apply spatial mask
            X_dynamic = np.multiply(scale_mask, X_stable) #power increases across channels
            X += X_dynamic #concatenate
        assert(len(freqs) == len(growth_decay_list))
        self.check_frequency_dynamics(dmd, kwargs, X, freqs, rtol, growth_decay_list)

