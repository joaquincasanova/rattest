import scipy.io as sio
import scipy.signal as sig
import scipy.fftpack as sfft
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

pi = np.pi
f0 = 60.
Q = 30.
K = 466./257.*1e6
plotit=False

def meg_rat_loc(n):
    r = 16#mm
    dtheta = pi - pi*(n-2)/n
    theta=np.arange(0,n-2)*dtheta
    x = r*np.cos(theta)
    y = 0
    z = r*np.sin(theta)
    return np.hstack((x,y,z))


def ecog_rat_loc(n):
    r = 16#mm
    dtheta = pi - pi*(n-2)/n
    theta=np.arange(0,n-2)*dtheta
    x = r*np.cos(theta)
    y = 0
    z = r*np.sin(theta)
    return np.hstack((x,y,z))

directory='./oceanit/05162017/'
names=['MEG_ECOG','ECOG','ECOG_MEG_DistX2','MEG_NoSpeaker','MEG_Speaker']

for name in names:
    data=sio.loadmat(directory+name+'.mat')

    StS1_start = np.array(data[name]['epocs'][0][0]['Wap_'][0][0]['onset'][0][0])
    StS1_stop_ = np.array(data[name]['epocs'][0][0]['Wap_'][0][0]['offset'][0][0])
    StS1_stop = StS1_start+1.0

    start = np.array(data[name]['epocs'][0][0]['Tick'][0][0]['onset'][0][0])
    stop = np.array(data[name]['epocs'][0][0]['Tick'][0][0]['offset'][0][0])
    
    StS1_=data[name]['snips'][0][0]['StS1'][0][0]['data'][0][0]
    rows = StS1_.shape[0]

    StS1 = StS1_[0:rows:2,:]+StS1_[1:rows+1:2,:]
    fs_StS1=data[name]['snips'][0][0]['StS1'][0][0]['fs'][0][0]

    ECoG = np.array(data[name]['streams'][0][0]['ECoG'][0][0]['data'][0][0])
    MEG = np.array(data[name]['streams'][0][0]['Meg1'][0][0]['data'][0][0])

    fs_ECoG = np.array(data[name]['streams'][0][0]['ECoG'][0][0]['fs'][0][0])
    fs_MEG = np.array(data[name]['streams'][0][0]['Meg1'][0][0]['fs'][0][0])

    b_ECoG, a_ECoG = sig.iirnotch(f0/(fs_ECoG/2),Q)
    b_MEG, a_MEG = sig.iirnotch(f0/(fs_MEG/2),Q)

    MEG_filt = sig.filtfilt(b_MEG,a_MEG,MEG,axis=1)
    ECoG_filt = sig.filtfilt(b_ECoG,a_ECoG,ECoG,axis=1)
    for m in range(1,6):
        b_ECoG, a_ECoG = sig.iirnotch(m*f0/(fs_ECoG/2),Q)
        b_MEG, a_MEG = sig.iirnotch(m*f0/(fs_MEG/2),Q)

        MEG_filt = sig.filtfilt(b_MEG,a_MEG,MEG_filt,axis=1)
        ECoG_filt = sig.filtfilt(b_ECoG,a_ECoG,ECoG_filt,axis=1)
    
    time_ECoG = np.cumsum(np.ones(ECoG.shape),1)/fs_ECoG
    time_MEG = np.cumsum(np.ones(MEG.shape),1)/fs_MEG

    level = np.array(data[name]['epocs'][0][0][0][0][1]['data'][0][0])
    
    n_epochs=start.shape[0]
    n_trials=StS1_start.shape[0]
    
    ECoG_epochs_index = []
    MEG_epochs_index = []
    
    ECoG_epoched = []
    MEG_epoched = []
    ECoG_fft_epoched = []
    MEG_fft_epoched = []
    ECoG_PSD_epoched = []
    MEG_PSD_epoched = []
    
    ECoG_filt_epoched = []
    MEG_filt_epoched = []
    ECoG_filt_fft_epoched = []
    MEG_filt_fft_epoched = []
    ECoG_filt_PSD_epoched = []
    MEG_filt_PSD_epoched = []

    f_ECoG = []
    f_MEG = []
    
    for i in range(0,n_trials):
        picks = np.where(time_ECoG[0,:]>=StS1_start[i])
        picks = np.intersect1d(picks,np.where(time_ECoG[0,:]<StS1_stop[i]))

        ECoG_epochs_index.append(picks)
        ECoG_epoched.append(ECoG[:,ECoG_epochs_index[i]])
        ECoG_filt_epoched.append(ECoG_filt[:,ECoG_epochs_index[i]])
        
        ECoG_fft_epoched.append(sfft.fftshift(sfft.fft(ECoG_epoched[i])))
        ECoG_filt_fft_epoched.append(sfft.fftshift(sfft.fft(ECoG_filt_epoched[i])))
        ECoG_PSD_epoched.append(10.*np.log10(np.abs(ECoG_fft_epoched[i]*np.conj(ECoG_fft_epoched[i]))))
        ECoG_filt_PSD_epoched.append(10.*np.log10(np.abs(ECoG_filt_fft_epoched[i]*np.conj(ECoG_filt_fft_epoched[i]))))
        n =ECoG_epoched[i].shape[1]
        f_ECoG.append(sfft.fftshift(sfft.fftfreq(n,1./fs_ECoG)))
        
        picks = np.where(time_MEG[0,:]>=StS1_start[i])
        picks = np.intersect1d(picks,np.where(time_MEG[0,:]<StS1_stop[i]))

        MEG_epochs_index.append(picks)
        MEG_epoched.append(MEG[:,MEG_epochs_index[i]])
        MEG_filt_epoched.append(MEG_filt[:,MEG_epochs_index[i]])
        n =MEG_epoched[i].shape[1]
        f_MEG.append(sfft.fftshift(sfft.fftfreq(n,1./fs_MEG)))
        
        MEG_fft_epoched.append(sfft.fftshift(sfft.fft(MEG_epoched[i])))
        MEG_filt_fft_epoched.append(sfft.fftshift(sfft.fft(MEG_filt_epoched[i])))
        MEG_PSD_epoched.append(10.*np.log10(np.abs(MEG_fft_epoched[i]*np.conj(MEG_fft_epoched[i]))))
        MEG_filt_PSD_epoched.append(10.*np.log10(np.abs(MEG_filt_fft_epoched[i]*np.conj(MEG_filt_fft_epoched[i]))))
            
        if plotit:
            plt.subplot(2,4,1)
            plt.plot(time_ECoG[:,ECoG_epochs_index[i]].T,ECoG_epoched[i].T)
            plt.title('ECoG Trial '+str(i)+' Level '+str(level[i])+' dB')

            plt.subplot(2,4,2)
            plt.plot(time_MEG[:,MEG_epochs_index[i]].T,MEG_epoched[i].T)
            plt.title('MEG Trial '+str(i)+' Level '+str(level[i])+' dB')

            plt.subplot(2,4,5)
            plt.plot(time_ECoG[:,ECoG_epochs_index[i]].T,ECoG_filt_epoched[i].T)
            plt.title('ECoG Filt Trial '+str(i)+' Level '+str(level[i])+' dB')

            plt.xlabel('Time (s)')
            
            plt.subplot(2,4,6)
            plt.plot(time_MEG[:,MEG_epochs_index[i]].T,MEG_filt_epoched[i].T)
            plt.title('MEG Filt Trial '+str(i)+' Level '+str(level[i])+' dB')

            plt.xlabel('Time (s)')
            
            plt.subplot(2,4,3)
            plt.plot(f_ECoG[i].T,ECoG_PSD_epoched[i].T)
            plt.title('ECoG Trial '+str(i)+' Level '+str(level[i])+' dB')
            plt.xlim(0,300)
            
            plt.subplot(2,4,4)
            plt.plot(f_MEG[i].T,MEG_PSD_epoched[i].T)
            plt.title('MEG Trial '+str(i)+' Level '+str(level[i])+' dB')
            plt.xlim(0,300)
            
            plt.subplot(2,4,7)
            plt.plot(f_ECoG[i].T,ECoG_filt_PSD_epoched[i].T)
            plt.title('ECoG Filt Trial '+str(i)+' Level '+str(level[i])+' dB')
            plt.xlim(0,300)
            
            plt.xlabel('Frequency (Hz)')
            
            plt.subplot(2,4,8)
            plt.plot(f_MEG[i].T,MEG_filt_PSD_epoched[i].T)
            plt.title('MEG Filt Trial '+str(i)+' Level '+str(level[i])+' dB')
            plt.xlim(0,300)
            
            plt.xlabel('Frequency (Hz)')
            plt.show()

    ECoG_levels = []
    MEG_levels = []
    
    ECoG_fft_levels = []
    MEG_fft_levels = []
        
    ECoG_PSD_levels = []
    MEG_PSD_levels = []

    i=0
    for l in [-120,0,5,10,15]:

        ECoG_trim = []
        MEG_trim = []
        ns_o = 1e6
        ms_o = 1e6

        picks, picksJ=np.where(level==l)

        for p in picks:
            ns = ECoG_filt_epoched[p].shape[1]
            if ns<ns_o:
                ns_o=ns

        for p in picks:
            ECoG_trim.append(ECoG_filt_epoched[p][:,0:ns_o])

        for p in picks:
            ms = MEG_filt_epoched[p].shape[1]
            if ms<ms_o:
                ms_o=ms

        for p in picks:
            MEG_trim.append(MEG_filt_epoched[p][:,0:ms_o])

        ECoG_levels.append(np.mean(np.array(ECoG_trim),0))
        MEG_levels.append(np.mean(np.array(MEG_trim),0))

        time_ECoG = np.arange(0,ns_o)/fs_ECoG

        time_MEG = np.arange(0,ms_o)/fs_MEG

        f_ECoG=(sfft.fftshift(sfft.fftfreq(ns_o,1./fs_ECoG)))
        
        f_MEG=(sfft.fftshift(sfft.fftfreq(ms_o,1./fs_MEG)))
        
        ECoG_fft_levels.append(sfft.fftshift(sfft.fft(ECoG_levels[i])))
        ECoG_PSD_levels.append(10.*np.log10(np.abs(ECoG_fft_levels[i]*np.conj(ECoG_fft_levels[i]))))
        
        MEG_fft_levels.append(sfft.fftshift(sfft.fft(MEG_levels[i])))
        MEG_PSD_levels.append(10.*np.log10(np.abs(MEG_fft_levels[i]*np.conj(MEG_fft_levels[i]))))
        
        plt.subplot(2,2,1)
        plt.plot(time_ECoG.T,ECoG_levels[i].T)
        plt.title('ECoG Level '+str(level[i])+' dB')

        plt.subplot(2,2,3)
        plt.plot(time_MEG.T,MEG_levels[i].T)
        plt.title('MEG Level '+str(level[i])+' dB')

        plt.xlabel('Time (s)')

        plt.subplot(2,2,2)
        plt.plot(f_ECoG.T,ECoG_PSD_levels[i].T)
        plt.title('ECoG Level '+str(level[i])+' dB')
        plt.xlim(0,300)

        plt.subplot(2,2,4)
        plt.plot(f_MEG.T,MEG_PSD_levels[i].T)
        plt.title('MEG Level '+str(level[i])+' dB')
        plt.xlim(0,300)

        plt.xlabel('Frequency (Hz)')
        plt.savefig('level'+str(i)+name+'.png')
        plt.close()

        i+=1
xyz = rat_loc(4)

