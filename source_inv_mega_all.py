#inversion for gaussian sources
import obspy
import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter,filtfilt,lfilter,sosfiltfilt,sosfilt
from scipy import signal
from scipy.optimize import nnls
import datetime
import pandas as pd
from scipy.linalg import norm
from l1ls import l1ls as L1solver
from matplotlib.path import Path
from math import ceil

def GMD_solve(G,D):
    #calculate inversion here
    GT=np.transpose(G)
    M=np.dot(np.dot(np.linalg.inv(np.dot(GT,G)),np.transpose(G)),D)
    return M

def GMD_solve_reg(G,D,alpha=np.logspace(-2,2,20),solver=1,L1=False,rel_tol=0.01):
    #calculate inversion here
    G=G.copy()
    D=D.copy()
    sav_misfit=[]
    sav_M=[]
    sav_Lm=[]
    if L1==False:
        #add regularization here
        reg=np.zeros([G.shape[1],G.shape[1]])
        for i in range(reg.shape[0]):
            reg[i,i]=1
        zd=np.zeros([G.shape[1],1])
        #loop all the alpha
        for a in alpha:
            new_reg=reg*a #append this under the G matrix
            new_G=np.vstack([G,new_reg])
            new_D=np.vstack([D,zd])
            #convert new_D to 1-D vector
            new_D.shape=(-1)
            if solver==1:
                M=GMD_solve(new_G,new_D)
                model=np.dot(G,M.reshape(len(M),1))
                misfit=np.sqrt(sum((model-D)**2))
                sav_M.append(M)
                sav_misfit.append(misfit)
                sav_Lm.append(norm(M))
            elif solver==2:
                M,rnorm=nnls(new_G,new_D)
                #get the regulization M and use it to the original G*M=Dhat
                model=np.dot(G,M.reshape(len(M),1))
                misfit=np.sqrt(sum((model-D)**2))
                sav_M.append(M)
                sav_misfit.append(misfit[0])
                sav_Lm.append(new_reg.dot(M))
            elif solver==3: #truncated SVD
                H=(G.T).dot(G)
                Gstar=np.linalg.pinv(H,rcond=a)
                Ginv=Gstar.dot(G.T)
                M=Ginv.dot(D)
                model=np.dot(G,M.reshape(len(M),1))
                misfit=np.sqrt(sum((model-D)**2))
                sav_M.append(M)
                sav_misfit.append(misfit)
                sav_Lm.append(norm(M))

        sav_misfit=np.array(sav_misfit)
        sav_Lm=np.array(sav_Lm)
        
    else:
        for a in alpha:
            [M, status, hist] = L1solver(G, D, a, tar_gap=rel_tol)
            model=np.dot(G,M.reshape(len(M),1))
            misfit=np.sqrt(sum((model-D)**2))
            sav_M.append(M)
            sav_misfit.append(misfit[0])
            sav_Lm.append(sum(abs(M))) #L1 norm
    return sav_misfit,sav_M,sav_Lm


def lowpass(data,fcorner,fsample,order,zerophase=True):

    if np.size(fcorner)==2:
        ftype='bandpass'
    else:
        ftype='lowpass'
    fnyquist=fsample/2.0
    sos = butter(order, np.array(fcorner)/(fnyquist),ftype,output='sos')
    #padding zero on the edge of data, careful if the boundary is not zero
    #pz=np.zeros(int(len(data)*1.0))
    #pzdata=np.concatenate([pz,data,pz])
    pzdata=data.copy()
    if zerophase==True:
        data_filt=sosfiltfilt(sos,pzdata)
    else:
        data_filt=sosfilt(sos,pzdata)
        
    data_filt -= data_filt[0]

    return data_filt



def read_gauge(filename,reftime):
    time=np.c_[np.genfromtxt(filename,usecols=0,dtype='float',skip_header=0)] #read time for the raw data
    datastart=datetime.datetime(2018,1,1)+datetime.timedelta(days=time[0][0]-1) #for raw data
    t1=[]
    for i,ti in enumerate(time):
        t1.append(datetime.datetime(2018,1,1)+datetime.timedelta(days=ti[0]-1)) #for raw data
    #Make every time relative ins econds to t1
    t=np.zeros(len(time))
    for k in range(len(time)):
        t[k]=(t1[k]-reftime).total_seconds()
    #read data
    data=np.genfromtxt(filename,usecols=1,dtype='float',skip_header=0)
    return(t,data)


def read_DART_and_trim(filename,reftime,cut_time,sampl):
    
    st=obspy.read(filename)
    st[0].trim(starttime=reftime+cut_time[0],endtime=reftime+cut_time[1])
    
    return st[0].data







##data path, depending on what the data look like, you may modify this part###
inversion_out='/Users/seansantellanes/Documents/Shumagin2020/ext_mega_inv_2/inv_results/'
# inversion_name='all_TSVD_path_filtered'
inversion_name='minari'
L1=False
min_amr=3
scalar=1

load_G=False
G_name='/Users/seansantellanes/Documents/Shumagin2020/ext_mega_inv_2/PICHU/G_check.npy'

#Three station vienrsion
stations=['/Users/seansantellanes/Documents/Shumagin2020/DART/46402modeling.sac',
          '/Users/seansantellanes/Documents/Shumagin2020/DART/46409modeling.sac',
          '/Users/seansantellanes/Documents/Shumagin2020/DART/46414modeling.sac',
          '/Users/seansantellanes/Documents/Shumagin2020/King_Cove/King_Cove_3.sac',
          '/Users/seansantellanes/Documents/Shumagin2020/Sand_Point/Sand_Point_3.sac',
          ]

syn_stations=['gauge46402.txt',
              'gauge46409.txt',
              'gauge46414.txt',
              'gauge90002.txt',
              'gauge90001.txt',
              ]

isDART=[True,
        True,
        True,
        False,
        False,
        ] #use read_gauge or read_DART function

cut_time=[(int(3600*0.15),int(3600*1.15)),#was 7185
          (int(3600*0.6),int(3600*1.5)),
          (int(3600*0.25),int(3600*1.2)),
          (int(3600*0),int(3600*1.95)), #was 7200 
          (int(3600*0),int(3600*1.69)), #Was 6230
          ]

reftime = obspy.UTCDateTime('2020-10-19T20:54:39') #earthquake OT time as reference (0 sec)
time_shift=np.load('/Users/seansantellanes/Documents/Shumagin2020/ext_mega_inv_2/time_shift.npy')            #0*60 # in seconds

tsum_bandpass = [np.array([1./7200,1./300]),  
                    np.array([1./7200,1./300]),
                    np.array([1/7200,1/300]),
                    np.array([1./7200,1./(60*2)]),  #12min because tide gauge sample rate is 6min
                    np.array([1./7200,1./(60*2)]),
                    ]

weight=True
sta_weights=[250, #More weight = more important 46402 was 80 before 06/18/2021 1:30 PM
          250, #was 300
          275, #was 150
          125, #was 200
          200,
          ]
#Three station vienrsion
# stations=[
#           '/Users/seansantellanes/Documents/Shumagin2020/DART/46409modeling.sac',
#           '/Users/seansantellanes/Documents/Shumagin2020/DART/46414modeling.sac',
#           '/Users/seansantellanes/Documents/Shumagin2020/King_Cove/King_Cove_3.sac',
#           '/Users/seansantellanes/Documents/Shumagin2020/Sand_Point/Sand_Point_3.sac',
#           ]

# syn_stations=[
#               'gauge46409.txt',
#               'gauge46414.txt',
#               'gauge90002.txt',
#               'gauge90001.txt',
#               ]

# isDART=[
#         True,
#         True,
#         False,
#         False,
#         ] #use read_gauge or read_DART function

# cut_time=[#was 7185
#           (int(3600*0.75),int(3600*1.6)),
#           (int(3600*0.32),int(3600*1.42)),
#           (int(3600*1.32),int(3600*1.741)), #was 7200 int(3600*1.347)
#           (int(3600*1.022),int(3600*1.714)), #Was 6230 int(3600*1.022)
#           ]

# reftime = obspy.UTCDateTime('2020-10-19T20:54:39') #earthquake OT time as reference (0 sec)
# time_shift=0*60 # in seconds

# tsum_bandpass = [  
#                     np.array([1./7200,1./300]),
#                     np.array([1/7200,1/300]),
#                     np.array([1./7200,1./(60*2)]),  #12min because tide gauge sample rate is 6min
#                     np.array([1./7200,1./(60*2)]),
#                     ]

# weight=True
# sta_weights=[ #More weight = more important 46402 was 80 before 06/18/2021 1:30 PM
#           300, #was 300
#           150, #was 150
#           150, #was 200
#           300,
#           ]
#filter along path
tsun_path=np.genfromtxt('/Users/seansantellanes/Documents/Shumagin2020/ext_mega_inv_2/PICHU/Thrust.txt')
grid=np.genfromtxt('/Users/seansantellanes/Documents/Shumagin2020/ext_mega_inv_2/info/grid.info')
P=Path(tsun_path)
filter_grids=True

zeroweight=1e5
zeroweight_activate=False
strictlyzeros=[np.arange(0,117),
                  np.arange(0,195),
                  np.arange(0,105),
                  np.arange(0,295),
                  np.arange(0,260)
                  ]



# inversion_out='/Users/dmelgarm/SandPoint2020/mega_inversion2/model_results/'
# # inversion_name='all_TSVD_path_filtered'
# inversion_name='tidegauge_L2'
# L1=False
# min_amr=3
# scalar=20

# load_G=False
# G_name='/Users/dmelgarm/SandPoint2020/mega_inversion2/geoclaw/G_all.npy'

# #Three station vienrsion
# stations=['/Users/dmelgarm/SandPoint2020/tide_gauges/king.sac',
#           '/Users/dmelgarm/SandPoint2020/tide_gauges/sand.sac'
#           ]

# syn_stations=['gauge09000.txt',
#               'gauge09001.txt'
#               ]

# isDART=[False,
#         False
#         ] #use read_gauge or read_DART function

# cut_time=[(0,6030),
#           (0,5925)
#           ]

# reftime = obspy.UTCDateTime('2020-10-19T20:54:39') #earthquake OT time as reference (0 sec)
# time_shift=0*60 # in seconds

# tsum_bandpass = [np.array([1./7200,1./720]),
#                  np.array([1./7200,1./720])
#                     ]

# weight=True
# sta_weights=[1,
#              1
#           ]

# ##filter along path
# tsun_path=np.genfromtxt('/Users/dmelgarm/SandPoint2020/null_space_L1_path4.1.txt')
# grid=np.genfromtxt('/Users/dmelgarm/SandPoint2020/mega_inversion2/info/grid.info')
# P=Path(tsun_path)
# filter_grids=True

# zeroweight_activate=True
# zeroweight=1e2
# strictlyzeros=[np.arange(0,295),
#                np.arange(0,260)
#                  ]




####   Run the ivnersion

sampl = 15 #resample to 15 sec
D = [] #Merged Data for inversion
sav_new_t = [] #save the individual time for events, so that you'll know that's the D corresponding for


###======Make subplots, merge all the tcs data into D for later inversion======###
station_weights=[]

for n,sta in enumerate(stations):
   
    #---read the data---
    data = read_DART_and_trim(sta,reftime,cut_time[n],sampl)

        
    #noramlzie by norm and by weight
    if weight==True:
        data_norm = norm(data)
        
        station_weight = (1/data_norm) * sta_weights[n]
        station_weights.append(station_weight)
        
        #Make weights matrix
        if n==0:
            w=np.ones(len(data))*station_weight
            if zeroweight_activate:
                wsz=np.ones(len(data))
                wsz[strictlyzeros[n]] = zeroweight
        else:
            w=np.r_[w,np.ones(len(data))*station_weight]
            if zeroweight_activate:
                wsz_temp=np.ones(len(data))
                wsz_temp[strictlyzeros[n]] = zeroweight
                wsz=np.r_[wsz,wsz_temp]
                        
    
    #---concatenate the data to a big array D for later inversion---
    try:
        D = np.hstack([D,data])
    except:
        D = data.copy()





###======Read synthetic tsunami data from gausian source========
##Read tsunami Syn data



Dir_Syns=np.sort(glob.glob('/Users/seansantellanes/Documents/Shumagin2020/ext_mega_inv_2/output/gauss*'))
# Dir_Syns=Dir_Syns[0:509]
all_syn=[]

dt_shift=15 #controls sampling of shifted time series
if load_G==True:
    all_syn=np.load(G_name)
else:
    n_grid=0
    dt_shift=15
    kshift=0
    for Dir_syn in Dir_Syns:
        
        #read synthetic data, and filter
        # print('load:',Dir_syn)
        temp_syn=[]
        
        # #check if is in filte grid
        # if filter_grids==True:
        #     run_GF=P.contains_point(grid[n_grid,:])
        #     n_grid += 1
        # else:   
        #     run_GF=True
        
        run_GF=True
        
        if run_GF:
            for n,syn_station in enumerate(syn_stations):
                
                syn_path = Dir_syn+'/'+'_output/'+syn_station
                syn = np.genfromtxt(syn_path,dtype='float')
                syn_time = syn[:,1]
                syn_data = syn[:,5]
                syn_amr = syn[:,0]
                
                #filter by amr
                if isDART[n]==False: #check for min_amr
                    iamr=np.where(syn_amr>=min_amr)[0]
                    syn_time=syn_time[iamr]
                    syn_data=syn_data[iamr]
                    
                #Apply scalar correction
                syn_data /= scalar
                
                #Add time shift to synthetics
                print(time_shift[kshift],int(time_shift[kshift]/dt_shift),ceil(time_shift[kshift]/dt_shift))
                Npts_added = int(ceil(time_shift[kshift]/dt_shift))
                syn_time=np.r_[np.arange(0,time_shift[kshift],dt_shift),syn_time+time_shift[kshift]]
                syn_data=np.r_[np.zeros(Npts_added),syn_data]
                
                #---interpolate the data, remove mean---
                interp_syn_t = np.arange(0,cut_time[n][1],sampl)  #new time for interpolate, make it long for the filter
                interp_syn_data = np.interp(interp_syn_t,syn_time,syn_data)
                interp_syn_data = interp_syn_data
                
                #---filter the syn data---
                # filt_syn_data = lowpass(interp_syn_data,tsum_bandpass[n],1.0/sampl,4,zerophase=True)
                filt_syn_data = interp_syn_data.copy()
                #---cut the data as the desired length---
                N=int((cut_time[n][1]-cut_time[n][0])/sampl)+1
                new_syn_t = np.linspace(cut_time[n][0],cut_time[n][1],N) 
                # new_syn_t = np.arange(cut_time[n][0],cut_time[n][1],sampl)  #use the same cut time as data
                new_syn_data = np.interp(new_syn_t,interp_syn_t,filt_syn_data)
                idx_org_time = np.where(np.abs(new_syn_t)==np.min(np.abs(new_syn_t)))[0][0]
                
                #---concatenate the synthetic data to a big matrix for later inversion---
                try:
                    temp_syn = np.hstack([temp_syn,new_syn_data])
                except:
                    temp_syn = new_syn_data.copy()
            
            #---make temp_syn be a N by 1 array---
            temp_syn.shape = (len(temp_syn),1)
            
            #---finally, save all the temp_syn (from every gaussian source) into all_syn
            try:
                all_syn = np.hstack([all_syn,temp_syn])
            except:
                all_syn = temp_syn.copy()
        kshift+=1
    np.save(G_name,all_syn)
    
    #Note: the shape of all_syn is now N (same as D length) by M (number of gaussian sources)


###====== Do inversion here=======
#make subpltios outside path be zero
run_GF=np.where(P.contains_points(grid)==False)[0]
for k in range(len(run_GF)):
    z=np.zeros((1,all_syn.shape[1]))
    z[0,run_GF[k]]=1
    z*=1e4 #strenght of regularization
    all_syn=np.r_[all_syn,z]
    D=np.r_[D,np.array(0)]


#add model cosntrin from AC12=-8cm of subsidence
z=np.zeros((1,all_syn.shape[1]))
# print(z, len(z[0]))
z[:]=1
z*=1e4 #strenght of regularization
all_syn=np.r_[all_syn,z]
D=np.r_[D,np.array(-0.15)]

L = np.logspace(-1,2,20)
# L = np.logspace(-8,-4,50) #for svd
RL = np.array(range(len(L)))

#---try different regularization params and plot misfit function---
print('Running inversions')


#Finalw eights matrix
W=np.diag(np.r_[w,np.ones(len(run_GF)+1)])

if zeroweight_activate:
    Wsz=np.diag(wsz)
    W=W.dot(Wsz)
D.shape=(len(D),1) #make sure the D is N by 1

#apply weights before inverting
Wd=W.dot(D)
WG=W.dot(all_syn)

sav_misfit,sav_M,sav_Lm = GMD_solve_reg(WG,Wd,alpha=L,solver=1,L1=L1) #solver=1 regular inversion, 2 nnls
   
#save to file
np.savetxt(inversion_out+inversion_name+'.model_norms',sav_Lm,fmt='%.4e')
np.savetxt(inversion_out+inversion_name+'.reg_values',L,fmt='%.4e')
np.savetxt(inversion_out+inversion_name+'.misfits',sav_misfit,fmt='%.4e')
np.savetxt(inversion_out+inversion_name+'.data',D,fmt='%.4e')
np.savetxt(inversion_out+inversion_name+'.data_weights',np.array(station_weights),fmt='%.4e')
for k in range(len(sav_M)):
    inv_num=str(k).rjust(4,'0')
    np.savetxt(inversion_out+inversion_name+'.inv'+inv_num,sav_M[k],fmt='%.4e')
    # print(inversion_out+inversion_name+'.inv'+inv_num,sav_M[k])
   
    # plt.plot(RL,sav_misfit,'-k',linewidth=1.5)
    # for i in RL:
    #     plt.text(RL[i],sav_misfit[i],str(RL[i]))
    # #model=np.dot(mergeG,M)
    # #M_filt=GMD_solve(mergeG_filt,D_filt)
    # #model_filt=np.dot(mergeG_filt,M_filt)
    # #misfit=sum((model-D_filt)**2)


# plt.plot(RL,sav_misfit,'r.')
# plt.ylabel('Misfit',fontsize=14)
# plt.xlabel('Smoothness',fontsize=14)
# plt.show()


# #---select a final model and plot model v.s. data---
# plt.figure()
# #if the 50th regularization is the preferred model
# best_id = 5
# model = np.dot(all_syn,sav_M[best_id])
# plt.plot(D,'k')
# plt.plot(model,'r')
# plt.legend(['Data','Model'])
# plt.show()
