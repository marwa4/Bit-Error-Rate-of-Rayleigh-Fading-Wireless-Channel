import numpy as np
import matplotlib.pyplot as plt
import numpy.random as nr

BlockLength = 10000 # transmitted symbols
Nblcoks = 1000 # Number of epoch
SNRdB = np.arange(1.0,45.0,3.0) # SNR range in dB
BER = np.zeros(SNRdB.size) # Initialization of the BER vector
BERth = np.zeros(SNRdB.size) # Initialization of the theoretical BER vector
SNR = 10**(SNRdB/10) # from dB to linear

for epoch in range(Nblcoks):
    h = (nr.normal(0.0,1.0,BlockLength)+1j*nr.normal(0.0,1.0,BlockLength))/np.sqrt(2) # generation of the Rayleigh coefficient
    Sym = 2 * nr.randint(2, size=BlockLength) - 1  # generation of BPSK symbols (-1, 1) the average power is P = 1
    noise = nr.normal(0.0, 1.0, BlockLength)+ 1j* nr.normal(0.0, 1.0, BlockLength)  # Generation of white noise with 0 mean and noise power  1
    for itersnrdB in range(SNRdB.size):
        Txbits = np.sqrt(SNR[itersnrdB])*Sym # different SNRs symbol
        Rxbits = h*Txbits+noise # multiply by the Rayleigh and add noise
        Decbits = 2*(np.real(np.conj(h)*Rxbits)>0)-1 # detection with threshold 0
        BER[itersnrdB] = BER[itersnrdB]+np.sum(Decbits!=Sym) # count number of bit errors


BER = BER/BlockLength/Nblcoks # over the number of total bits
BERth = 1/2*(1-np.sqrt(SNR/(2+SNR))) # Average BER for the Rayleigh wireless channel


plt.yscale('log')
plt.plot(SNRdB, BER, 'go')
plt.plot(SNRdB, BERth,'r-') # Qfunction for Pe (cdf = 1-qf)
plt.grid(1,which = 'both')
plt.suptitle('BER for Rayleigh Fadding Channel')
plt.xlabel('SNR (dB)')
plt.ylabel('BER')
plt.legend(['Experiments', 'Theory'])
plt.show()
