import scipy.io as sio
import scipy.signal as sig
import scipy.fftpack as sfft
import numpy as np
import pandas
import h5py
import pickle
import os.path

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

pi = np.pi
f0 = 60.#mains
fcu = 10.#cutoff
fcl = .2#cutoff
Q = 30.#notch Q at 60, bw=2hz
q=48#decimation
K = 466./257.*1e6
plotit=True
lowpass=True#lowpass is redundant with decimate's built in butterworth low-pass
highpass=True
mains = True
def spec(x,fs,name,sensor,label):
    f, t, Sxx = sig.spectrogram(x, fs, nperseg = max(64,int(x.shape[0]/8)))
    F,T = np.meshgrid(f,t)
    plt.pcolormesh(T,F,Sxx.T)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.savefig(name+'spec'+'label'+str(label)+'sensor'+str(sensor)+'.png')

def trials_corr(data):
    trials = data.shape[0]
    sensors = data.shape[1]
    steps = data.shape[2]
    corr_ = np.zeros([trials*(trials-1)/2,sensors,steps])
    print "Trials ",trials,"Sensors ",sensors,"Steps ",steps,"Comparisons ",trials*(trials-1)/2
    for k in range(0,sensors):
        nn=0
        for i in range(0,trials):
            for j in range(i+1,trials):
                corr_[nn,k,:]=sig.correlate(data[i,k,:],data[j,k,:],mode='same')
                nn+=1
    corr_mean = np.mean(corr_,axis=0)
    corr_std = np.std(corr_,axis=0)
    return corr_std, corr_mean

def plotcorr(ECoG_corr_std,ECoG_corr_mean,MEG_corr_std,MEG_corr_mean,fs_MEG,fs_ECoG,label, name):
    time_ECoG = np.arange(0,ECoG_corr_mean.shape[1])/fs_ECoG

    time_MEG = np.arange(0,MEG_corr_mean.shape[1])/fs_MEG
    plt.subplot(2,2,1)
    plt.plot(time_ECoG.T,ECoG_corr_mean.T)
    plt.title('ECoG Corr Mean Label '+str(label))
    plt.autoscale(enable=True, axis='y', tight=True)

    plt.subplot(2,2,3)
    plt.plot(time_MEG.T,MEG_corr_mean.T)
    plt.title('MEG Corr Mean Label '+str(label))
    plt.ylim(-1e-3,1e-3)#autoscale(enable=True, axis='y', tight=True)    
    plt.xlabel('Time (s)')

    plt.subplot(2,2,2)
    plt.plot(time_ECoG.T,ECoG_corr_std.T)
    plt.title('ECoG Corr Std Label '+str(label))    
    plt.autoscale(enable=True, axis='y', tight=True)

    plt.subplot(2,2,4)
    plt.plot(time_MEG.T,MEG_corr_std.T)
    plt.title('MEG Corr Std Label '+str(label))
    plt.autoscale(enable=True, axis='y', tight=True)
    plt.xlabel('Time (s)')
    plt.savefig(name+'label'+str(label)+'.png')
    plt.close()
    
def plot3d(data,xyz,batch,step):

    fig = plt.figure()
    
    print "plotting real" 
    fplot=np.squeeze(data[batch][:,step])

    ax = fig.add_subplot(1,1,1, projection='3d')

    ax.plot_trisurf(np.squeeze(xyz[:,1]), np.squeeze(xyz[:,2]), fplot, cmap=cm.jet, linewidth=0.2)

    plt.show()

def plot_fft(MEG,ECoG,fs_MEG,fs_ECoG,label,name):
    print 'MEG: ',MEG.shape
    print 'ECoG: ',ECoG.shape
    axlim=('std' not in name)
    time_ECoG = np.arange(0,ECoG.shape[1])/fs_ECoG

    time_MEG = np.arange(0,MEG.shape[1])/fs_MEG
    
    f_ECoG=(sfft.fftshift(sfft.fftfreq(ECoG.shape[1],1./fs_ECoG)))
        
    f_MEG=(sfft.fftshift(sfft.fftfreq(MEG.shape[1],1./fs_MEG)))

    h_ECoG = np.matlib.repmat(sig.hann(ECoG.shape[1]),ECoG.shape[0],1)

    h_MEG = np.matlib.repmat(sig.hann(MEG.shape[1]),MEG.shape[0],1)
    
    ECoG_fft = sfft.fftshift(sfft.fft(ECoG*h_ECoG))
    ECoG_PSD = 10.*np.log10(np.abs(ECoG_fft*np.conj(ECoG_fft)))

    MEG_fft = sfft.fftshift(sfft.fft(MEG*h_MEG))
    MEG_PSD = 10.*np.log10(np.abs(MEG_fft*np.conj(MEG_fft)))

    plt.subplot(2,2,1)
    plt.plot(time_ECoG.T,ECoG.T)
    plt.title('ECoG Label '+str(label))
    if axlim: plt.ylim(-150e-6,150e-6)
    if not axlim: plt.autoscale(enable=True, axis='y', tight=True)
    
    plt.subplot(2,2,3)
    plt.plot(time_MEG.T,MEG.T)
    plt.title('MEG Label '+str(label))
    if axlim: plt.ylim(-2e-9,2e-9)
    if not axlim: plt.autoscale(enable=True, axis='y', tight=True)
    plt.xlabel('Time (s)')

    plt.subplot(2,2,2)
    plt.plot(f_ECoG.T,ECoG_PSD.T)
    plt.title('ECoG Label '+str(label))
    plt.xlim(0,300)
    if axlim: plt.ylim(-120.,-40.)
    if not axlim: plt.autoscale(enable=True, axis='y', tight=True)
    
    plt.subplot(2,2,4)
    plt.plot(f_MEG.T,MEG_PSD.T)
    plt.title('MEG Label '+str(label))
    plt.xlim(0,300)
    if axlim: plt.ylim(-220.,-140.)
    if not axlim: plt.autoscale(enable=True, axis='y', tight=True)


    plt.xlabel('Frequency (Hz)')
    plt.savefig(name+'label'+str(label)+'.png')
    plt.close()

def remove_stim(data,time,start,stop_):
    print 'Remove stimulus'
    mask = []
    for i in range(0,start.shape[0]):
        picks = np.where(time>=start[i])
        picks = np.intersect1d(picks,np.where(time<stop_[i]))
        mask = np.union1d(mask,picks)
    data_ = np.delete(data,mask,1)
    time_ = np.delete(time,mask,1)
    print data.shape, time.shape
    print data_.shape, time_.shape
    return data_, time_
    
def sph2cart(az,el,r):
    x = r * np.cos(el) * np.cos(az)
    y = r * np.cos(el) * np.sin(az)
    z = r * np.sin(el)
    return x, y, z
    
def meg_rat_loc():
    r = 16.#mm
    dtheta = pi/3
    theta=np.array([0.,1.,2.,3.])
    x = r*np.cos(theta)
    y = [0.,0.,0.,0.]
    z = r*np.sin(theta)
#    print 'x ',x
#    print 'y ',y
#    print 'z ',z
    return np.hstack((x,y,z)).reshape(3,4).T


def ecog_rat_loc():
    p = 0.750#mm
    x0, y0 = -1.0,0.0
    #15 16 13 14
    # 9 12 11 10
    # 7  6  5  8
    # 1  2  3  4
    #xp,yp on cortical surface
    xp = np.array([-3*p,-2*p,  -p,   0,
            -p,-2*p,-3*p,   0,
          -3*p,   0,  -p,-2*p,
            -p,   0,-3*p,-2*p])+x0
    yp = np.array([   0,   0,   0,   0,
             p,   p,   p,   p,
           2*p, 2*p, 2*p, 2*p,
           3*p, 3*p, 3*p, 3*p])+y0
    r=14.*np.ones(np.size(yp))#mm
    phi = np.arcsin(yp/r)
    theta = np.arccos(xp/(r*np.cos(phi)))
    y,x,z=sph2cart(theta,phi,r)
#    print 'x ',x
#    print 'y ',y
#    print 'z ',z
    return np.hstack((x,y,z)).reshape(3,16).T

#    return np.hstack((x.T,y.T,z.T))

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
        n_treat=levels_in.shape[0]
    else:        
        treatments[:,0]=level[sequence[0:len(sequence)]]
        treatments[:,1]=freq[sequence[0:len(sequence)]]
        treatments[:,2]=sequence[0:len(sequence)]
        n_treat=len(level)
    return treatments, n_treat
                    
directories=['./oceanit/05162017/','./oceanit/05182017/',   './oceanit/05182017/', './oceanit/05182017/','./oceanit/05182017/','./oceanit/05182017/']
names      =[           'MEG_ECOG',        'ECOG_MEG_P1','ECOG_Live_1_Bad_ground',         'ECOG_Live_2',     'ECOG_MEG_Tones', 'ECOG_MEG_Iso_Tones']#
PinkFile='Oceanit1'
ToneFile='Oceanit2'

for p in range(0,len(names)):
    name=names[p]
    directory=directories[p]
    
    if not os.path.isfile(name+'.pickle'): 
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
        else:
            csvname_seq = directory+PinkFile+'.seq.csv'
            csvname_par = directory+PinkFile+'.par.csv'
            period = 1.0#s
            flag = 'P1'
        if directory is './oceanit/05162017/':
            flag = 'P0'

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

        except Exception, e:
            print e
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

#        ECoG, time_ECoG = remove_stim(ECoG,time_ECoG,start,stop_)
#        MEG, time_MEG = remove_stim(MEG,time_MEG,start,stop_)

        #Read sequence data    
        treatments,n_treat = read_sequence(csvname_seq,csvname_par,flag,level)
        print 'Filter entire waveforms'
        if q==1 or fs_MEG0<1500.:
            MEG_dec=MEG
            fs_MEG = fs_MEG0
        else:
            MEG_dec = np.zeros([MEG.shape[0],int(MEG.shape[1]/q)+1])        
            print 'Decimate+anti-aliasing'
            for i in range(0,MEG.shape[0]):
                MEG_dec[i,:] = sig.decimate(MEG[i,:], q, n=None, ftype='iir', zero_phase=True)
            fs_MEG = fs_MEG0/q

        if q==1 or fs_ECoG0<1500.:
            ECoG_dec=ECoG
            fs_ECoG = fs_ECoG0
        else:
            ECoG_dec = np.zeros([ECoG.shape[0],int(ECoG.shape[1]/q)+1])
            print 'Decimate+anti-aliasing'
            for i in range(0,ECoG.shape[0]):
                ECoG_dec[i,:] = sig.decimate(ECoG[i,:], q, n=None, ftype='iir', zero_phase=True) 
            fs_ECoG = fs_ECoG0/q
        
        if lowpass:
            print 'Low-pass filter ', fcu
            b_ECoG_lp, a_ECoG_lp =sig.butter(8,fcu/(fs_ECoG/2),btype='lowpass')
            b_MEG_lp, a_MEG_lp =sig.butter(8,fcu/(fs_MEG/2),btype='lowpass')
            MEG_lp = sig.filtfilt(b_MEG_lp,a_MEG_lp,MEG_dec,axis=1)
            ECoG_lp = sig.filtfilt(b_ECoG_lp,a_ECoG_lp,ECoG_dec,axis=1)
        else:
            MEG_lp=MEG_dec
            ECoG_lp=ECoG_dec
        
        if highpass:
            print 'High-pass filter ',fcl
            b_ECoG_hp, a_ECoG_hp =sig.butter(2,fcl/(fs_ECoG/2),btype='highpass')
            b_MEG_hp, a_MEG_hp =sig.butter(2,fcl/(fs_MEG/2),btype='highpass')
            MEG_hp = sig.filtfilt(b_MEG_hp,a_MEG_hp,MEG_lp,axis=1)
            ECoG_hp = sig.filtfilt(b_ECoG_hp,a_ECoG_hp,ECoG_lp,axis=1)
        else:
            MEG_hp=MEG_lp
            ECoG_hp=ECoG_lp            

        if mains:
            print 'Notch filter ',f0,Q
            b_ECoG, a_ECoG = sig.iirnotch(f0/(fs_ECoG/2),Q)
            b_MEG, a_MEG = sig.iirnotch(f0/(fs_MEG/2),Q)

            MEG_filt = sig.filtfilt(b_MEG,a_MEG,MEG_hp,axis=1)
            ECoG_filt = sig.filtfilt(b_ECoG,a_ECoG,ECoG_hp,axis=1)
            for m in range(2,6):
                print 'Notch filter ',f0*m,Q*m
                #print 'Notch filter ',m*f0/(fs_ECoG/2)
                b_ECoG, a_ECoG = sig.iirnotch(m*f0/(fs_ECoG/2),Q*m)
                b_MEG, a_MEG = sig.iirnotch(m*f0/(fs_MEG/2),Q*m)

                MEG_filt = sig.filtfilt(b_MEG,a_MEG,MEG_filt,axis=1)
                ECoG_filt = sig.filtfilt(b_ECoG,a_ECoG,ECoG_filt,axis=1)
        else:
            MEG_filt=MEG_hp
            ECoG_filt=ECoG_hp      
        
 
        time_ECoG = np.arange(0,ECoG_filt.shape[1])/fs_ECoG
        time_MEG = np.arange(0,MEG_filt.shape[1])/fs_MEG
       
        print 'Epoch filtered trials'
        n_epochs=start.shape[0]

        ECoG_epochs_index = []
        MEG_epochs_index = []

        ECoG_filt_epoched = []
        MEG_filt_epoched = []

        for i in range(0,n_epochs):
            picks = np.where(time_ECoG>=start[i])
            picks = np.intersect1d(picks,np.where(time_ECoG<stop[i]))

            ECoG_filt_picks = ECoG_filt[:,picks]

            picks = np.where(time_MEG>=start[i])
            picks = np.intersect1d(picks,np.where(time_MEG<stop[i]))

            MEG_filt_picks = MEG_filt[:,picks]


            MEG_filt_epoched.append(MEG_filt_picks)
            ECoG_filt_epoched.append(ECoG_filt_picks)

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
        ECoG_3=np.array(ECoG_trim)
        MEG_3=np.array(MEG_trim)

        print 'Group by treatment.'

        with open(name+'.pickle', 'w') as f:
            pickle.dump({"ECoG_3":ECoG_3, "MEG_3":MEG_3, "fs_MEG":fs_MEG, "fs_ECoG":fs_ECoG, "flag":flag, "n_treat":n_treat, "treatments":treatments}, f)
        del data, MEG, ECoG
    else:
        with open(name+'.pickle', 'r') as f:
            b=pickle.load(f)
            ECoG_3 = b["ECoG_3"]
            MEG_3 = b["MEG_3"]
            fs_MEG = b["fs_MEG"]
            fs_ECoG = b["fs_ECoG"]
            flag = b["flag"]
            n_treat = b["n_treat"]
            treatments = b["treatments"]
            
    MEG_average = []
    ECoG_average = []

    MEG_std = []
    ECoG_std = []
    if flag=='Tones' or flag=='P0':
        indices = range(1,n_treat+1)
    elif flag=='P1':
        indices = [-120.,-20.,-15.,-10.,-5.,0.,5.]
        
    ll=0
    for l in indices:
        picks = np.where(treatments[:,2]==l)
        picks=picks[0]
        print 'Treatment: ', l,'Level indices: ', picks
            
        if picks.shape==0:
            continue
        else:
            tmp = np.mean(MEG_3[picks,:,:],axis=0)/K
            MEG_average.append(tmp)
            tmp = np.mean(ECoG_3[picks,:,:],axis=0)
            ECoG_average.append(tmp)
            
            tmp = np.std(MEG_3[picks,:,:],axis=0)/K
            MEG_std.append(tmp)
            tmp = np.std(ECoG_3[picks,:,:],axis=0)
            ECoG_std.append(tmp)
            
            #ECoG_corr_std, ECoG_corr_mean = trials_corr(ECoG_3[picks,:,:])
            #MEG_corr_std, MEG_corr_mean = trials_corr(MEG_3[picks,:,:])
            if plotit:
                plot_fft(MEG_average[ll],ECoG_average[ll],fs_MEG,fs_ECoG,ll,name+'mean')
                #plot_fft(MEG_std[ll],ECoG_std[ll],fs_MEG,fs_ECoG,ll,name+'std')
                #plotcorr(ECoG_corr_std,ECoG_corr_mean,MEG_corr_std,MEG_corr_mean,fs_MEG,fs_ECoG,ll, name+'corr')
                #for sensor in range(0,MEG_average[ll].shape[0]):
                #    spec(MEG_average[ll][sensor,:],fs_MEG,name+'MEG',sensor,ll)
                #for sensor in range(0,ECoG_average[ll].shape[0]):
                #    spec(ECoG_average[ll][sensor,:],fs_ECoG,name+'ECoG',sensor,ll)    
        ll+=1
        meg_xyz = meg_rat_loc()
        if flag=='P0':
            meg_xyz = np.flipud(meg_xyz)
        ecog_xyz = ecog_rat_loc()
#        plot3d(ECoG_average,ecog_xyz,ll-1,300)
        with open(name+'.grouped.pickle', 'w') as f:
            pickle.dump({"ECoG_average":ECoG_average, "MEG_average":MEG_average, "fs_MEG":fs_MEG, "fs_ECoG":fs_ECoG, "flag":flag, "n_treat":n_treat, "treatments":treatments, "meg_xyz":meg_xyz, "ecog_xyz":ecog_xyz}, f)#MEG in Tesla
