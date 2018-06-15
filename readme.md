
# Dynamic Mode Decomposition (DMD) and Multi-Resolution DMD (mrDMD)
Decoding hand movements from ECoG recordings

## Installation
```Bash
git clone https://github.com/BruntonUWBio/ecog-hand
sudo apt-get install python3-pip
sudo pip3 install numpy matplotlib cvxpy pytest sklearn
```
## Usage

In your project, load the required modules:


```python
%matplotlib inline
from mrDMD import mrDMD
from DMD import DMD
from helper_functions import *
```

Start with a signal composed of a sum of sinusoids in 1-dimension


```python
dt = 1/200
N = 1000
t = np.linspace(0, 5, N)
amp = 1
freq_max = 40
freqs = np.arange(freq_max)
freqs = freqs[::4]
print('Freqs: ')
print(freqs)

X = buildX(freqs, t)
plt.figure()
plt.plot(t, X[0,:])
plt.show()
```

    Freqs: 
    [ 0  4  8 12 16 20 24 28 32 36]



![png](output_3_1.png)


Comparisons to FFT when frequency well below Nyquist


```python
freq, P = fftPlot(X, dt, freq_max)
```


![png](output_5_0.png)



```python
stack_factor = 2*len(freqs)
kwargs = {'dt':dt, 
    'scale_modes':True}
dmd = DMD(**kwargs)
dmd.fit(X)
f, P = dmd.spectrum(sort='frequencies')

idx = in_range(f, (1,freq_max))
plt.figure()
plt.stem([f[i] for i in idx], [P[i] for i in idx])
plt.title('DMD spectrum shortened')
plt.xlabel('Frequency')
plt.show()
plt.close()
```


![png](output_6_0.png)


## Testing

Pytest is used for testing run the following at the command line

```Bash
pytest
```

## References
For DMD algorithm details see:
* "Dynamic Mode Decomposition". J. Nathan Kutz, Steven L. Brunton, Bingni W. Brunton, and Joshua L. Proctor 2016.

For multiresolution DMD algorithm details see:
* "Multiresolution Dynamic Mode Decomposition". J. Nathan Kutz, Xing Fu, Steven L. Brunton 2016.
