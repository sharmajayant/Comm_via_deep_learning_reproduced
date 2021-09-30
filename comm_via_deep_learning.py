from __future__ import division
import argparse
import gc
from time import time
import os
import sys
from tqdm import tqdm

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(123)
np.random.seed(123)

import torch.distributions as distributions

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_args():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--exp-dir',type=str)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden-size', type=int, default=50)
    parser.add_argument('--rate',type=float,default=2)
    parser.add_argument('--memory',type=int, default= 3, help = '# Memory elements used in TCM enc.')
    parser.add_argument('--train-snr',type=float,default=0.0, help = 'Train snr for neural decoder.')
    parser.add_argument('--test-snr-low',type=float,default=0.0 , help='Lower value of test range.')
    parser.add_argument('--test-snr-high',type=float,default=13.0,help='Higher value of test range')
    parser.add_argument('--test-snr-step',type=float,default=0.5)
    parser.add_argument('--numblocks',type=int,default=50000,help='# Codewords used for training' )
    parser.add_argument('--batch-size',type=int,default=1000, help='# Codewords per batch')
    parser.add_argument('--timesteps',type=int,default=100,help='Length of each codeword.')
    parser.add_argument('--variant',type=str, default='lstm', choices={'rnn', 'gru', 'lstm'})
    parser.add_argument('--directions', type = str, default='bi', choices={'uni', 'bi'})
    parser.add_argument('--num-hidden-layers', type=int, default=2, help='# Stacked recurrent layers.')
    parser.add_argument('--noise-type', type=str, default= 'gaussian')
    parser.add_argument('--train-mode',type=str,default='fix')
    parser.add_argument('--blocklen',type=int,default=100)
    
    args = parser.parse_args()
    
    return args

print('Using device: ', device)

def noise_sigma(snr_val):
    std         =   10**(-snr_val/20)
    return std

class gaussian_channel(nn.Module):    
    def __init__(self, noise_snr, mode, snr_low=0.0,snr_high=10.0):
        super(gaussian_channel,self).__init__()
        self.noise_mean         = 0
        self.noise_std          = noise_sigma(noise_snr)
        self.snr_low            = snr_low
        self.snr_high           = snr_high
        self.mode               = mode
        print('Channel Description :')
        print('Noise Type : Gaussian')
        
        if mode == 'single_snr':
            print('Noise_SNR  : {}dB'.format(noise_snr))
            print('Noise_mean : {}'.format(self.noise_mean))
            print('Noise_std  : {}'.format(self.noise_std))
        elif mode == 'range':
            print('Mode             : range')
            print('Noise_mean       : {}'.format(self.noise_mean))
            print('Noise_snr_range  : {}dB - {}dB'.format(self.snr_low,self.snr_high))
    
    def forward(self, x):
        
        if self.mode == 'range':
            snr_gen                 =   distributions.uniform.Uniform(self.snr_low,self.snr_high)
            noise_snr               =   snr_gen.sample()
            noise_std               =   noise_sigma(noise_snr)
            
        elif self.mode == 'fix':
            noise_std               =   self.noise_std
        
        self.noise_generator    = distributions.normal.Normal(self.noise_mean,noise_std)
        noise                   = self.noise_generator.sample(x.shape).double().to(device)
        output                  = x + noise
        
        return output



class codeword_dataset(Dataset):
    
    def __init__(self, numblocks=5000,rate = 2, generator_matrix = np.array([[1,1,1],[1,0,1]]),feedback=np.array([1,1,1]), blocklen=100, memory = 3):
        
        # Only rate 1/n supported in this implementation
        # Only BPSK shown here  modulation_type = 'BPSK'
        
        self.generator_matrix   =   generator_matrix
        self.rate               =   rate
        assert generator_matrix.shape[0] == rate 
        assert generator_matrix.shape[1] == memory 
        
        self.blocklen           =   args.blocklen
        self.modulation_type    =   'BPSK'  
        self.numblocks          =   numblocks
        self.feedback           =   feedback
        
        self.x             =    []
        self.y             =    []
        
        for __ in range(numblocks):
            bits            =   np.random.randint(0,2,(self.blocklen))
            
            modulated_msg   =   []
            for stream_num in range(self.rate): 
                temp        =   np.convolve(bits, self.generator_matrix[stream_num]) %2
                temp        =   2*temp- 1
                
                temp        =   temp[:-(memory-1)]
                
                modulated_msg.append(temp)
            
            
            modulated_msg   =   np.array(modulated_msg)
            self.x.append(modulated_msg.transpose())
            self.y.append(bits.transpose())
            
        self.x          =   np.array(self.x).astype(float)
        self.y          =   np.array(self.y).astype(float)
        
    
    def __getitem__(self,idx):
        return self.x[idx], self.y[idx]
    
    def __len__(self):
        return self.numblocks

def train(epoch_number):
    
    dec_obj.train()
    train_loss          =   0.0
    correct             =   0.0
    
    channel_obj         =   gaussian_channel(args.train_snr,'fix')
    
    channel_obj.double()
    channel_obj.to(device)
    
    with tqdm(enumerate(train_loader), total = len(train_loader)) as batch_iterator:
        for idx,(x,y) in batch_iterator:
            
            optimizer.zero_grad()
            x,y         =   x.to(device), y.to(device)
            channel_out =   channel_obj(x)
            out_dec     =   dec_obj(channel_out)
            loss        =   criterion(out_dec, y)
            
    #        print("Loss Value : {}".format(loss))
            loss.backward()
            optimizer.step()
            
            train_loss                     +=   loss.item()
            temp                            =   out_dec.round().detach()
            
            correct                        +=   torch.sum(temp==y.detach()).to('cpu').numpy()
        
    train_acc       =   float(correct)/(len(train_dataset)* args.blocklen)
    str_print       =   "Epoch Number {:}: \t Train Loss = {:}, \t\t Train BER = {:}".format(epoch_number, train_loss, 1.0-train_acc)
        
    print(str_print)
    
    gc.collect()
    
    return train_loss, train_acc


def test(snr,chan_type='gaussian'):
    
    dec_obj.eval()
    test_loss           =   0.0
    correct             =   0.0
    
    if chan_type == "gaussian":
        channel_obj         =   gaussian_channel(snr,'fix')

    channel_obj.double()
    channel_obj.to(device)
    
    with torch.no_grad():
        for idx,(x,y) in enumerate(test_loader):
        
            x,y             =   x.to(device), y.to(device)
            
            channel_out     =   channel_obj(x)
            out_dec         =   dec_obj(channel_out)
            loss            =   criterion(out_dec,y)
            test_loss      +=   loss.item()
            
            temp                            =   out_dec.round().detach().to(device)
            correct                        +=   torch.sum(temp==y.detach()).to('cpu').numpy()
            
    test_acc            =   float(correct)/(len(test_dataset)* args.blocklen)
    str_print           =   "                  \t Test  Loss = {:f}, \t\t Test BER  = {:}".format(test_loss,1.0-test_acc)
    
    print(str_print)
    
    gc.collect()
    
    return test_loss,test_acc

def test_in_range(chan_type='gaussian'):
    
    l                   =   args.test_snr_low
    h                   =   args.test_snr_high + args.test_snr_step
    step                =   args.test_snr_step
    snr_range           =   np.arange(l, h, step)
    
    test_ber_vals       =   []
    test_snr_vals       =   []
    
    for curr_snr in snr_range:
        print('Currrent SNR value : {}'.format(curr_snr))
        __, acc_val     =   test(curr_snr, chan_type)
        
        test_ber_vals.append(1.0-acc_val)
        test_snr_vals.append(curr_snr)
    
    test_ber_vals       =   np.array(test_ber_vals)
    test_snr_vals       =   np.array(test_snr_vals)
    
    return test_snr_vals, test_ber_vals


def lrscheduler(epoch):
    
    some_num    =   10
    factor      =   2
    
    if epoch<some_num+1:
        lr = args.lr
    elif epoch<2*some_num+1:
        lr = args.lr/(factor**1)
    elif epoch<3*some_num+1:
        lr = args.lr/(factor**2)
    elif epoch<4*some_num+1:
        lr = args.lr/(factor**3)
    elif epoch<5*some_num+1:
        lr = args.lr/(factor**4)
    elif epoch<6*some_num+1:
        lr = args.lr/(factor**5)
    elif epoch<7*some_num+1:
        lr = args.lr/(factor**6)
    elif epoch<8*some_num+1:
        lr = args.lr/(factor**7)
    elif epoch<9*some_num+1:
        lr = args.lr/(factor**8)
    elif epoch<10*some_num+1:
        lr = args.lr/(factor**9)
    else:
        lr = args.lr/(factor**10)
    
    print('\nlr = {}\n'.format(lr))
    
    return lr


class neural_dec(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, directions, variant, num_hidden_layers):
        super(neural_dec,self).__init__()
        self.hidden_size        =   hidden_size
        self.output_size        =   output_size
        
        self.input_size         =   input_size
        self.num_layers         =   num_hidden_layers
        
        if directions == 'bi':
            self.bi     =   True
        else:
            self.bi     =   False
        
        if variant.lower() == 'rnn':
            self.recc_layers  = nn.RNN(input_size,hidden_size,num_layers=self.num_layers,batch_first=True,bidirectional=self.bi)
        elif variant.lower() == 'gru':
            self.recc_layers  = nn.GRU(input_size,hidden_size,num_layers=self.num_layers,batch_first=True,bidirectional=self.bi)
        elif variant.lower() == 'lstm':
            self.recc_layers  = nn.LSTM(input_size,hidden_size,num_layers=self.num_layers,batch_first=True,bidirectional=self.bi)
        else:
            print('Invalid variant. Exiting.')
            sys.exit()
            
        
        if self.bi:
            self.lin1                =   nn.Linear(2*self.hidden_size,self.output_size)
        else:
            self.lin1                =   nn.Linear(self.hidden_size,self.output_size)
        
        self.activation =   nn.Sigmoid()
        
        
    def forward(self,x):
        shape                   =   x.shape
        output, __              =   self.recc_layers(x)
        output                  =   self.lin1(output)
        output                  =   self.activation(output)
        output                  =   output.squeeze()
        
        return output


if __name__ == '__main__':
    
    tic             =   time()
    args            =   get_args()
                                                           #                  1+D+D2   1 + D2
#    numblocks=5000,rate = 2, generator_matrix = np.array([[1,1,1],[1,0,1]]), memory = 3
    train_dataset   =   codeword_dataset(args.numblocks, args.rate, np.array([[1,1,1],[1,0,1]]), args.blocklen, args.memory)
    train_loader    =   DataLoader(train_dataset,shuffle=True, batch_size=args.batch_size, num_workers = 0)
    
    test_dataset    =   codeword_dataset(args.numblocks, args.rate, np.array([[1,1,1],[1,0,1]]), args.blocklen, args.memory)
    test_loader     =   DataLoader(test_dataset,shuffle=True, batch_size=args.batch_size, num_workers = 0)
    
                                   # input_size, hidden_size, output_size, directions, variant, num_hidden_layers
    dec_obj         =   neural_dec(args.rate, args.hidden_size, 1, args.directions, args.variant, args.num_hidden_layers).double().to(device)   
    
    channel_obj     =   gaussian_channel(0.0,args.train_snr)
    
#    dec_obj         =   nn.DataParallel(dec_obj)
    
    criterion       =   nn.BCELoss()  # nn.MSELoss()
    
    lr              =   args.lr
    
    for epoch in range(1, args.epochs + 1):
#            lr          =   lrscheduler(epoch)
        optimizer   =   optim.Adam(list(dec_obj.parameters()), lr=lr)
        train(epoch)
        test(args.train_snr)
        gc.collect()
    
    
    test_snr_vals, test_ber_vals    =   test_in_range()
    
    write_these     =   {}
    write_these['SNR_value ']   =   test_snr_vals
    write_these['BER_value ']   =   test_ber_vals
    
    filename        =   os.path.join('results','snr_and_ber'+'.csv')
    rng_test_file   =   open(filename,'w')
    
    df=pd.DataFrame(data=write_these,dtype=np.float32)
    df.to_csv(rng_test_file)
    rng_test_file.close()
    
    toc             =   time()
    
    print('Total runtime is {} seconds.'.format(toc-tic))
