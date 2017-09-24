import numpy as np
from matplotlib import pylab as plt

in_range = lambda x, lim: [i for i, v in enumerate(x) if (v >=
    lim[0]) and (v <= lim[1])]

def scale1(arr, mval=None):
    if mval is None:
        mval = np.max(arr)
    return np.array([i / mval  for i in arr])

def scale_2d(mat):
    mval = np.max(mat)
    mat /= mval
    return mat

def day_plot(time_idx, x, title='', step=1, color='k',
        scale_sig=True, savefig=False):
    if time_idx is 'default':
        time_idx = list(range(len(x)))
    plt.figure()
    plt.rc('text', usetex=True)
    for ri, row in enumerate(x):
        current_step = ri * step
        y = x[ri, :]
        scaled = scale1(y)
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

def gaussian2(size, fwhm=3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """
    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]
    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]
    gauss =  np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)
    return gauss.reshape((size, size))

def build_spatial(size, t, freq_list, noise_amp=.05, plot=False, methods=[None, None]):
    N = len(t)
    #Create Signal
    test = np.ones((size,size))
    gauss = gaussian2(size, fwhm=size, center=(size,size))
    test = gauss.dot(test)
    test = gauss.dot(test.T)
    test = scale_2d(test)
    if plot:
        plt.figure()
        plt.pcolor(test)
        title = 'Corner spatial pattern (%d Hz)' % freq_list[0]
        plt.title(title)
        plt.xlabel('Channels'); plt.ylabel('Channels')
        #plt.savefig('%s.pdf' % title)
    gauss_right = test.reshape((1, size **2))

    gauss_mid = gaussian2(size, fwhm=size, center=(size//2,size//2))
    temp = np.ones((size, size))
    temp = temp.dot(gauss_mid)
    gauss_mid = temp.T.dot(gauss_mid.T)
    gauss_mid = scale_2d(gauss_mid)
    if plot:
        plt.figure()
        plt.xlabel('Channels'); plt.ylabel('Channels')
        title = 'Center spatial pattern (%d Hz)' % freq_list[1]
        plt.title(title)
        plt.pcolor(gauss_mid)
        plt.show()
        plt.savefig('%s.pdf' % title)
    gauss_mid = gauss_mid.reshape((1, size **2))

    freqs = [freq_list[0]]
    x, spec5, f5 = build_signal(freqs, t, size=size ** 2, phase=np.pi/2, spec=True,
                        method=methods[0])
    #heat(x)
    for i in range(N):
        x[:,i] = gauss_right * x[:,i]
        
    freqs = [freq_list[1]]
    x_mid, spec17, f17 = build_signal(freqs, t, size=size ** 2, phase=np.pi/5, spec=True,
                               method=methods[1])
    for i in range(N):
        x_mid[:,i] = gauss_mid * x_mid[:,i]
    x += x_mid
    x += noise_amp * np.random.rand(*x.shape) #noise

    assert(all(f5 == f17))
    if plot:
        plt.figure()
        plt.pcolor(spec5 + spec17, cmap='hot')
        plt.show()
    ideal_spec = spec5 + spec17
    if plot:
        day_plot(t, x, title='Signal', step=2)
    spatial_list = [gauss_right, gauss_mid]
    return x, ideal_spec, spatial_list

def fft_plot(sig, dt, freq_max, plot=True):
    sig = sig[0,:] # assume sig is single dimension for now
    ps = np.abs(np.fft.fft(sig)) ** 2
    freq = np.fft.fftfreq(sig.size, dt)
    idx = np.argsort(freq)
    freq = freq[idx]
    ps = ps[idx]

    #within range
    idx = in_range(freq, (0, freq_max))
    freq = freq[idx]
    ps = ps[idx]

    #plot
    if plot:
        plt.plot(freq, ps)
        plt.title('FFT power spectrum')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel(r'Power ($V^2$)')
        plt.show()
    return freq, ps

def build_signal(freqs, t, size=1, noise_amp=0, method=None,
        spec=False, dt=.01, freq_max=50, phase=0):
    X = np.zeros((size, len(t)))
    N = len(t)
    f = np.linspace(0, int(max(freqs)), int(max(freqs)))
    spectrum = np.zeros((len(f) + 1, N))
    if spec:
        f, P = fft_plot(X, dt, freq_max, plot=False)
        spectrum = np.zeros((len(f), N))
        for i in range(N):
            spectrum[:, i] = P 
    for freq in freqs:
        X[:, :] += sinusoid(freq, t, phase=phase)
        spectrum[int(freq),:] = 1
    if method is 'third':
        X[:, :(N/3)] = 0
        X[:, (2*N/3):] = 0
        if spec:
            spectrum[:, :(N/3)] = 0
            spectrum[:, (2*N/3):] = 0
    if method is 'lastThird':
        X[:, :(2*N/3)] = 0
        if spec:
            spectrum[:, :(2*N/3)] = 0
    if method is 'tenth':
        tenth_ind = N // 10
        X[:, tenth_ind:] = 0
        if spec:
            spectrum[:, tenth_ind:] = 0
    if method is 'middleTenth':
        tenth_ind = N // 10
        start_ind = 4 * tenth_ind
        end_ind = N // 2
        X[:, :start_ind] = 0
        X[:, end_ind:] = 0
        if spec:
            spectrum[:, :start_ind] = 0
            spectrum[:, end_ind:] = 0
    X += noise_amp * np.random.rand(*X.shape) #noise
    if spec:
        return X, spectrum, f
    else:
        return X
    
def sinusoid(freq, t, phase=0):
    amp = 1
    w = freq * (2 * np.pi)
    X = amp*np.sin(w*t + phase)
    X = X.reshape((1, len(t)))
    return X

def primes(n):
    """ Returns a list of primes < n """
    n = int(n)	
    sieve = [True] * n
    for i in np.arange(3, n ** 0.5 + 1, 2, dtype=int):
        if sieve[i]:
            sieve[i*i::2*i]=[False]*((n-i*i-1)//(2*i)+1)
    return [2] + [i for i in np.arange(3,n,2) if sieve[i]]

