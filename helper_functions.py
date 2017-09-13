import numpy as np
from matplotlib import pylab as plt

in_range = lambda x, lim: [i for i, v in enumerate(x) if (v >=
    lim[0]) and (v <= lim[1])]

def fftPlot(sig, dt, freq_max, plot=True):
    sig = sig[0,:] # assume sig is single dimension for now
    ps = np.abs(np.fft.fft(sig)) ** 2
    freq = np.fft.fftfreq(sig.size, dt)
    #ps = ps[0, :]
    #order by freq
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
        plt.ylabel('Power')
        plt.show()
    return freq, ps

def buildX(freqs, t, size=1, noise_amp=0, method=None,
        spec=False, dt=.01, freq_max=50, phase=0):
    X = np.zeros((size, len(t)))
    N = len(t)
    f = np.linspace(0, int(max(freqs)), int(max(freqs)))
    spectrum = np.zeros((len(f) + 1, len(t)))
    #if spec:
        #f, P = fftPlot(X, dt, freq_max, plot=False)
        #P = scale1(P)
        #for i in range(N):
            #spectrum[:, i] = P 
    for freq in freqs:
        X[:, :] += sinX(freq, t, phase=phase)
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
    
def sinX(freq, t, phase=0):
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

