import scipy.io as sio
import scipy.signal as sig
import scipy.fftpack as sfft
import numpy as np
import csv
import h5py

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

pi = np.pi
f0 = 60.
fc = 300.
gs = 20.
gp = 0.
Q = 30.
q=48
K = 466./257.*1e6
plotit=True

def meg_rat_loc(n):
    r = 16#mm
    dtheta = pi - pi*(n-2)/n
    theta=np.arange(n-1,-1,-1)*dtheta
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

def read_sequence(fname_seq,fname_par):
    csvfile = open(fname_seq,'r')
    try:
        reader=csv.reader(csvfile)
        rownum=0
        for row in reader:
            if rownum==0:
                header=row
            else:
                try:
                    print row
                except:
                    continue
    finally:
        csvfile.close()

    csvfile = open(fname_par,'r')
    try:
        reader=csv.reader(csvfile)
        rownum=0
        for row in reader:
            if rownum==0:
                header=row
            else:
                try:
                    print row
                except:
                    continue
    finally:
        csvfile.close()
                    
directory='./oceanit/05182017/'
names=['ECOG_MEG_Iso_Tones','ECOG_MEG_P1','ECOG_MEG_Tones']#'ECOG_Live_1_Bad_ground','ECOG_Lives_2_Pink',]
PinkFile='Oceanit1'
ToneFile='Oceanit2'

for name in names:
    if 'P1' in name:
        csvname_seq = PinkFile+'.seq.csv'
        csvname_par = PinkFile+'.par.csv'
        period = 1.0#s
    elif 'Tones' in name:
        csvname_seq = directory+ToneFile+'.seq.csv'
        csvname_par = directory+ToneFile+'.par.csv'
        period = 0.25#s
        
    #Read sequence data    
    #treatments = read_sequence(csvname_seq,csvname_par)

    #Load trials
    try:
        data=sio.loadmat(directory+name+'.mat')

        #start-stop_:sound plays
        #stop_-stop:response

        start = np.array(data[name]['epocs'][0][0]['Wap_'][0][0]['onset'][0][0])
        stop_ = np.array(data[name]['epocs'][0][0]['Wap_'][0][0]['offset'][0][0])
        stop = start+period

        ECoG = np.array(data[name]['streams'][0][0]['ECoG'][0][0]['data'][0][0])
        MEG = np.array(data[name]['streams'][0][0]['Meg1'][0][0]['data'][0][0])

        fs_ECoG = np.array(data[name]['streams'][0][0]['ECoG'][0][0]['fs'][0][0])
        fs_MEG = np.array(data[name]['streams'][0][0]['Meg1'][0][0]['fs'][0][0])

        time_ECoG = np.cumsum(np.ones(ECoG.shape),1)/fs_ECoG
        time_MEG = np.cumsum(np.ones(MEG.shape),1)/fs_MEG

    except:
        data = h5py.File(directory+name+'.mat')

        #start-stop_:sound plays
        #stop_-stop:response

        start = np.array(data[name]['epocs']['Wap_']['onset']).T
        stop_ = np.array(data[name]['epocs']['Wap_']['offset']).T
        stop = start+period

        ECoG = np.array(data[name]['streams']['ECoG']['data']).T
        MEG = np.array(data[name]['streams']['Meg1']['data']).T

        fs_ECoG = np.array(data[name]['streams']['ECoG']['fs'])
        fs_MEG = np.array(data[name]['streams']['Meg1']['fs'])

        time_ECoG = np.cumsum(np.ones(ECoG.shape),1)/fs_ECoG
        time_MEG = np.cumsum(np.ones(MEG.shape),1)/fs_MEG
        
    #Epoch trials/filter epoched trials
    n_epochs=start.shape[0]
    
    ECoG_epochs_index = []
    MEG_epochs_index = []
     
    ECoG_epoched = []
    MEG_epoched = []
     
    ECoG_filt_epoched = []
    MEG_filt_epoched = []
    
    for i in range(0,n_epochs):
        picks = np.where(time_ECoG[0,:]>=start[i])
        picks = np.intersect1d(picks,np.where(time_ECoG[0,:]<stop[i]))

        ECoG_epochs_index.append(picks)
        ECoG_epoched.append(ECoG[:,ECoG_epochs_index[i]])
        
        picks = np.where(time_MEG[0,:]>=start[i])
        picks = np.intersect1d(picks,np.where(time_MEG[0,:]<stop[i]))

        MEG_epochs_index.append(picks)
        MEG_epoched.append(MEG[:,MEG_epochs_index[i]])
        
        if plotit:
            plt.subplot(2,1,1)
            plt.plot(time_ECoG[:,ECoG_epochs_index[i]].T,ECoG_epoched[i].T)
            plt.title('ECoG Trial '+str(i)+' Level '+str(level[i])+' dB')

            plt.subplot(2,1,2)
            plt.plot(time_MEG[:,MEG_epochs_index[i]].T,MEG_epoched[i].T)
            plt.title('MEG Trial '+str(i)+' Level '+str(level[i])+' dB')

            plt.xlabel('Time (s)')
            plt.show()

        #Low-pass filter
        b_ECoG_lp, a_ECoG_lp =sig.iirdesign(fc/(fs_ECoG/2), fc/(fs_ECoG/2)*1.1, gpass, gstop)
        b_MEG_lp, a_MEG_lp =sig.iirdesign(fc/(fs_MEG/2), fc/(fs_MEG/2)*1.1, gpass, gstop)

        MEG_lp = sig.filtfilt(b_MEG_lp,a_MEG_lp,MEG_epoched[i],axis=1)
        ECoG_lp = sig.filtfilt(b_ECoG_lp,a_ECoG_lp,ECoG_epoched[i],axis=1)

        #Decimate+anti-aliasing
        MEG_dec = decimate(MEG_lp, q, n=None, ftype='iir', axis=1)
        ECoG_dec = decimate(ECoG_lp, q, n=None, ftype='iir', axis=1)

        fs_ECoG = fs_ECog/q
        fs_ECoG = fs_ECog/q

        #Notch filter
        b_ECoG, a_ECoG = sig.iirnotch(f0/(fs_ECoG/2),Q)
        b_MEG, a_MEG = sig.iirnotch(f0/(fs_MEG/2),Q)

        MEG_filt = sig.filtfilt(b_MEG,a_MEG,MEG_dec,axis=1)
        ECoG_filt = sig.filtfilt(b_ECoG,a_ECoG,ECoG_dec,axis=1)
        for m in range(1,6):
            b_ECoG, a_ECoG = sig.iirnotch(m*f0/(fs_ECoG/2),Q)
            b_MEG, a_MEG = sig.iirnotch(m*f0/(fs_MEG/2),Q)

            MEG_filt = sig.filtfilt(b_MEG,a_MEG,MEG_filt,axis=1)
            ECoG_filt = sig.filtfilt(b_ECoG,a_ECoG,ECoG_filt,axis=1)


        MEG_filt_epoched.append(MEG_filt)
        ECoG_filt_epoched.append(ECoG_filt)

    #Trim trials to same length.
    ECoG_trim = []
    MEG_trim = []
    ns_o = 1e6
    ms_o = 1e6

    for p in range(0,n_trials):
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

xyz = rat_loc(4)

