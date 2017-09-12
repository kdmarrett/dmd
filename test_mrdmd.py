#Author : Karl Marrett kdmarrett@gmail.com

import numpy as np
from helperFunctions import *
from DMD import DMD
from mrDMD import mrDMD
import pytest

class TestMRDMD:

    @pytest.fixture(scope="module")
    def mrdmd(self, request):
        """ Setup mrdmd fit on sine wave before any method of this 
        class"""
        dt = .01
        kwargs = {'dt':dt, 'stack_factor':'estimate',
                'scale_modes':True, 'use_optimal_SVHT':True}
        mrdmd = mrDMD(kwargs)
        yield mrdmd
        print("Tear down multi-resolution dmd instance")
        del mrdmd

    @pytest.fixture(scope="function")
    def mrdmd_fitted(self, request):
        """ Setup mrdmd fit on sine wave before any method of this 
        class"""
        N = 1000
        t = np.linspace(0, 10, N)
        freqs = [3.3]
        size = 1
        dt = .01
        x = buildX(freqs, t, size=size)
        kwargs = {'dt':dt, 'stack_factor':'estimate',
                'scale_modes':True, 'use_optimal_SVHT':True}
        mrdmd = mrDMD(kwargs)
        mrdmd.fit(x)
        yield mrdmd
        print("Tear down multi-resolution dmd instance")
        del mrdmd

    @pytest.mark.skip()
    def test_spectrum(self, mrdmd_fitted):
        #test freq_space
        freq_space = np.arange(10)
        f, P = mrdmd_fitted.spectrum(freq_space=freq_space)
        assert(len(f) == len(P))
        for fval in f:
            assert(fval in freq_space)

        #test sort power
        Phi_mu_lambdas = (mrdmd_fitted.Phi, mrdmd_fitted.mu, mrdmd_fitted.lambdas)
        f, P = mrdmd_fitted.spectrum(sort='power', sortModes=True)
        last = P[0]
        for pval in P:
            pval >= last
        Phi_mu_lambdas_final = (mrdmd_fitted.Phi, mrdmd_fitted.mu, mrdmd_fitted.lambdas)
        for match in zip(Phi_mu_lambdas, Phi_mu_lambdas_final):
            assert(match[0].shape == match[1].shape)

        #test sort frequencies
        f, P = mrdmd_fitted.spectrum(sort='frequencies')
        last = f[0]
        for fval in f:
            fval >= last
        Phi_mu_lambdas_final = (mrdmd_fitted.Phi, mrdmd_fitted.mu, mrdmd_fitted.lambdas)
        for match in zip(Phi_mu_lambdas, Phi_mu_lambdas_final):
            assert(match[0].shape == match[1].shape)

    @pytest.mark.skip()
    def test_fit(self, mrdmd_fitted):
        pass

    @pytest.mark.skip()
    def test_reconstruct(self, mrdmd):
        pass

    def test_scalar_add(self, mrdmd):
        upper_lim = 10
        size = 10
        value = 5
        full_arr = np.random.randint(upper_lim, size=size)
        scalar_add_arr = mrdmd._scalar_add(full_arr, value)
        for non_scaled, scaled in zip(full_arr, scalar_add_arr):
            assert(non_scaled + value == scaled)

    def test_scale1(self, mrdmd):
        upper_lim = 10
        size = 20
        full_arr = np.random.randint(upper_lim, size=size)
        scaled_arr = mrdmd._scale1(full_arr)
        assert(np.max(scaled_arr) == 1.0)
        known_max = np.max(full_arr)
        scaled_arr = mrdmd._scale1(full_arr, mval=2 * known_max)
        assert(np.max(scaled_arr) == .5)

    def test_in_range(self, mrdmd):
        lims = (0, 10)
        size = 10
        rand_list = np.random.randint(lims[0], lims[1], size)
        indices_in_range = mrdmd._in_range(rand_list, lims)
        # all in list are in range therefore # indices should # match
        assert(len(indices_in_range) == size)
        rand_list = [i for i in rand_list]
        rand_list.append(lims[1] + 1)
        indices_in_range = mrdmd._in_range(rand_list, lims)
        #should not include any more indices since out of range
        assert(len(indices_in_range) == size)

    @pytest.mark.skip()
    def test_record_power_and_phi(self, mrdmd_fitted):
        pass
        #original = mrdmd_fitted.spec

    @pytest.mark.skip()
    def check_frequency_dynamics(self, mrdmd, kwargs, X, freqs,
            rtol, growth_decay_list):
        """Test growth and decay of modes
        given signal of known dynamics"""
        mrdmd = DMD(**kwargs)
        mrdmd.fit(X)
        f, P = mrdmd.spectrum(sort='power', sortModes=True)
        last_expected_mode = 2 * len(freqs) 
        #examine only dominant modes
        lambdas = mrdmd.lambdas[-last_expected_mode:]
        P = P[-last_expected_mode:]
        f = f[-last_expected_mode:]
        for freq, growth_decay in zip(freqs, growth_decay_list):
            f_idx = mrdmd._find_nearest_idx(np.asarray(f), freq)
            lambda_i = lambdas[f_idx]
            if growth_decay is 'stable':
                assert(np.isclose(abs(lambda_i), 1.0))
            elif growth_decay is 'decay':
                assert(abs(lambda_i) < 1.0)
            elif growth_decay is 'growth':
                assert(abs(lambda_i) > 1.0)

    @pytest.mark.skip()
    def check_spatial_accuracy(self, mrdmd, kwargs, X, freqs, rtol):
        mrdmd = DMD(**kwargs)
        mrdmd.fit(X)
        #sort: highest power first
        f, P = mrdmd.spectrum(sort='power', sortModes=True, plotfig=False)
        # two modes for each actual signal
        last_expected_mode = 2 * len(freqs) 
        original_channels = X.shape[0]
        Phi = np.abs(mrdmd.Phi[:original_channels])
        #check majority power within expected modes
        assert(np.isclose(np.sum(Phi), np.sum(Phi[:, -last_expected_mode:]), rtol=.01))
        #assert each chan increasing power like input signal
        for i in range(last_expected_mode):
            last = 0
            for val in Phi[:, -(i + 1)]:
                #check values are nondecreasing for chans
                assert(val >= last)
                last = val

    @pytest.mark.skip()
    def check_spectrum_elements(self, mrdmd, kwargs, X, freqs, rtol):
        mrdmd = DMD(**kwargs)
        mrdmd.fit(X)
        #sort: highest power first
        f, P = mrdmd.spectrum(sort='power', plotfig=False)
        # two modes for each actual signal
        last_expected_mode = 2 * len(freqs) 

        #check that most dominant signals 
        # were contained in original signal
        for f_ind in f[-last_expected_mode:]:
            freqs_idx = mrdmd._find_nearest_idx(np.asarray(freqs), f_ind)
            assert(np.isclose(f_ind, freqs[freqs_idx], rtol=rtol))

        #check each original signal contained in final
        for freq in freqs:
            f_idx = mrdmd._find_nearest_idx(np.asarray(f), freq)
            assert(np.isclose(freq, f[f_idx], rtol=rtol))

    @pytest.mark.skip()
    def test_frequency_spectrum(self, mrdmd):
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
        self.check_spectrum_elements(mrdmd, kwargs, X, freqs, rtol)

        # Spatial test
        chans = 10
        X_multi = np.repeat(X, chans, axis=0) 
        # build spatial mask
        mask = np.diag([(chans ** -1) * i for i in range(chans)])
        #apply spatial mask
        X_spatial = mask.dot(X_multi) #power increases across channels
        self.check_spatial_accuracy(mrdmd, kwargs, X_spatial, freqs, rtol)

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
        self.check_frequency_dynamics(mrdmd, kwargs, X, freqs, rtol, growth_decay_list)

