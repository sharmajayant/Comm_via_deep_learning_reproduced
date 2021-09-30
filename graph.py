import numpy as np
import matplotlib.pyplot as plt

snr =   np.array([0. ,  0.5,  1. ,  1.5,  2. ,  2.5,  3. ,  3.5,  4. ,  4.5,  5. ,
        5.5,  6.0])

ber =   np.array([0.0927816,0.066574,0.0458024,0.0294772,0.0179088,0.010349,0.0055812,0.0028648,0.0013864,0.0006974,0.0003116,0.000131,0.0000652])

plt.grid('on')
plt.semilogy(snr,ber,marker='*')
plt.xlabel('SNR')
plt.ylabel('BER')

plt.savefig('graph.png')
