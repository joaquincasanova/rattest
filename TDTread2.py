import scipy.io as sio
import scipy.signal as sig
import scipy.fftpack as sfft
import numpy as np
import pandas
import h5py

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

pi = np.pi
f0 = 60.
fc = 300.
gs = 60.
gp = 1.
Q = 30.
q=48
K = 466./257.*1e6
plotit=False

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

def read_sequence(fname_seq,fname_par,flag,levels_in):
    print fname_seq, fname_par
    seq = pandas.read_csv(fname_seq)
    par = pandas.read_csv(fname_par)
    sequence = seq['Seq-1']
    print sequence
    level = par['WaveAmp (dB)']
    print level
    freq = par['WaveFreq (Hz)']
    print freq
    treatments = np.zeros([len(sequence),3])
    if flag is 'P1':
        treatments=np.hstack((levels_in,levels_in*0.0,levels_in))
        print 'Levels file is incorrect for P1 test, use levels embedded in real data.'
    else:        
        treatments[:,0]=level[sequence[0:len(sequence)]]
        treatments[:,1]=freq[sequence[0:len(sequence)]]
        treatments[:,2]=sequence[0:len(sequence)]

    n_treat=levels_in.shape[0]
    return treatments, n_treat
                    
directory='./oceanit/05182017/'
names=['ECOG_MEG_P1','ECOG_MEG_Tones','ECOG_MEG_Iso_Tones']#'ECOG_Live_1_Bad_ground','ECOG_Lives_2_Pink',]
PinkFile='Oceanit1'
ToneFile='Oceanit2'

for name in names:
    if 'P1' in name:
        csvname_seq = directory+PinkFile+'.seq.csv'
        csvname_par = directory+PinkFile+'.par.csv'
        period = 1.0#s
        flag = 'P1'
    elif 'Tones' in name:
        csvname_seq = directory+ToneFile+'.seq.csv'
        csvname_par = directory+ToneFile+'.par.csv'
        period = 0.25#s
        flag = 'Tones'
        
    print 'Load trials ',name
    try:
        print 'Try scipy.io'
        data=sio.loadmat(directory+name+'.mat')

        #start-stop_:sound plays
        #stop_-stop:response

        start = np.array(data[name]['epocs'][0][0]['Wap_'][0][0]['onset'][0][0])
        stop_ = np.array(data[name]['epocs'][0][0]['Wap_'][0][0]['offset'][0][0])
        stop = start+period

        ECoG = np.array(data[name]['streams'][0][0]['ECoG'][0][0]['data'][0][0])
        MEG = np.array(data[name]['streams'][0][0]['Meg1'][0][0]['data'][0][0])

        fs_ECoG0 = np.array(data[name]['streams'][0][0]['ECoG'][0][0]['fs'][0][0])
        fs_MEG0 = np.array(data[name]['streams'][0][0]['Meg1'][0][0]['fs'][0][0])

        time_ECoG = np.arange(0,ECoG.shape[1])/fs_ECoG0
        time_MEG = np.arange(0,MEG.shape[1])/fs_MEG0

        level = np.array(data[name][0][0]['epocs'][0][0]['Wap_'][0][0]['data'])
        
    except:
        print 'v7.3 - use h5py'
        data = h5py.File(directory+name+'.mat')

        #start-stop_:sound plays
        #stop_-stop:response

        start = np.array(data[name]['epocs']['Wap_']['onset']).T
        stop_ = np.array(data[name]['epocs']['Wap_']['offset']).T
        stop = start+period

        ECoG = np.array(data[name]['streams']['ECoG']['data']).T
        MEG = np.array(data[name]['streams']['Meg1']['data']).T

        fs_ECoG0 = np.array(data[name]['streams']['ECoG']['fs'])
        fs_MEG0 = np.array(data[name]['streams']['Meg1']['fs'])

        time_ECoG = np.arange(0,ECoG.shape[1])/fs_ECoG0
        time_MEG = np.arange(0,MEG.shape[1])/fs_MEG0

        level = np.array(data[name]['epocs']['Wap_']['data']).T

    #Read sequence data    
    treatments,n_treat = read_sequence(csvname_seq,csvname_par,flag,level)

    print 'Epoch trials/filter epoched trials'
    n_epochs=start.shape[0]
    
    ECoG_epochs_index = []
    MEG_epochs_index = []
     
    ECoG_epoched = []
    MEG_epoched = []
     
    ECoG_filt_epoched = []
    MEG_filt_epoched = []
    
    for i in range(0,n_epochs):
        picks = np.where(time_ECoG>=start[i])
        picks = np.intersect1d(picks,np.where(time_ECoG<stop[i]))

        ECoG_epoched = ECoG[:,picks]
        
        picks = np.where(time_MEG>=start[i])
        picks = np.intersect1d(picks,np.where(time_MEG<stop[i]))

        MEG_epoched = MEG[:,picks]
        
        print 'Low-pass filter'
        b_ECoG_lp, a_ECoG_lp =sig.iirdesign(fc/(fs_ECoG0/2), fc/(fs_ECoG0/2)*1.1, gp, gs)
        b_MEG_lp, a_MEG_lp =sig.iirdesign(fc/(fs_MEG0/2), fc/(fs_MEG0/2)*1.1, gp, gs)

        MEG_lp = sig.filtfilt(b_MEG_lp,a_MEG_lp,MEG_epoched,axis=1)
        ECoG_lp = sig.filtfilt(b_ECoG_lp,a_ECoG_lp,ECoG_epoched,axis=1)

        print 'Decimate+anti-aliasing'
        MEG_dec = sig.decimate(MEG_lp, q, n=None, ftype='iir', axis=1, zero_phase=True)
        ECoG_dec = sig.decimate(ECoG_lp, q, n=None, ftype='iir', axis=1, zero_phase=True)
        
        fs_ECoG = fs_ECoG0/q
        fs_MEG = fs_MEG0/q

        print 'Notch filter'
        b_ECoG, a_ECoG = sig.iirnotch(f0/(fs_ECoG/2),Q)
        b_MEG, a_MEG = sig.iirnotch(f0/(fs_MEG/2),Q)

        MEG_filt = sig.filtfilt(b_MEG,a_MEG,MEG_dec,axis=1)
        ECoG_filt = sig.filtfilt(b_ECoG,a_ECoG,ECoG_dec,axis=1)
        for m in range(2,6):
            #print 'Notch filter ',m*f0/(fs_ECoG/2)
            b_ECoG, a_ECoG = sig.iirnotch(m*f0/(fs_ECoG/2),Q)
            b_MEG, a_MEG = sig.iirnotch(m*f0/(fs_MEG/2),Q)

            MEG_filt = sig.filtfilt(b_MEG,a_MEG,MEG_filt,axis=1)
            ECoG_filt = sig.filtfilt(b_ECoG,a_ECoG,ECoG_filt,axis=1)


        MEG_filt_epoched.append(MEG_filt)
        ECoG_filt_epoched.append(ECoG_filt)

    print 'Trim trials to same length.'
    ECoG_trim = []
    MEG_trim = []
    ns_o = 1e6
    ms_o = 1e6

    for p in range(0,n_epochs):
        ns = ECoG_filt_epoched[p].shape[1]
        if ns<ns_o:
            ns_o=ns

    for p in range(0,n_epochs):
        ECoG_trim.append(ECoG_filt_epoched[p][:,0:ns_o])

    for p in range(0,n_epochs):
        ms = MEG_filt_epoched[p].shape[1]
        if ms<ms_o:
            ms_o=ms

    for p in range(0,n_epochs):
        MEG_trim.append(MEG_filt_epoched[p][:,0:ms_o])

    print 'Smush them into an array.'
    ECoG_3=np.array(ECoG_Trim)
    MEG_3=np.array(MEG_Trim)

    print 'Group by treatment.'

    #free up memory
    del MEG_lp,ECoG_lp,MEG,ECoG

    MEG_average = []
    ECoG_average = []
    if flag=='Tones':
        for l in range(0,n_treat):
            picks = np.where(treatments[2,:]==l)
            MEG_average.append(np.mean(MEG_3[picks,:,:],axis=0))
            ECoG_average.append(np.mean(ECoG_3[picks,:,:],axis=0))
    elif flag=='P1':
        for l in [-120,-20,-15,-10,-5,0,5]:
            picks = np.where(treatments[0,:]==l)
            MEG_average.append(np.mean(MEG_3[picks,:,:],axis=0))
            ECoG_average.append(np.mean(ECoG_3[picks,:,:],axis=0))
            
xyz = rat_loc(4)

