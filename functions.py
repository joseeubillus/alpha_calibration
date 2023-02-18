'''
    This file contains all the functions used for data processing and calibration, 
    if any change is needed please refer to this file.

    Authors: Hailun Ni and Ubillus Jose

'''
# Import require libraries

import numpy as np
import cv2
import scipy.io
import os
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.linear_model import LinearRegression

## Constructors
def constructor_drainage(start_image_dra,end_image_dra,steps_divider=50):

    mask_file=scipy.io.loadmat('Data_processed/BW.mat')
    mask=mask_file.get('BW')

    interval_dra = (end_image_dra-start_image_dra)//steps_divider   
    image_sequence_dra = np.arange(start=start_image_dra,stop=end_image_dra+interval_dra,step=interval_dra,dtype=int)
    num_image_dra = len(image_sequence_dra)

    Iw = cv2.imread('Data_processed/Drainage_processed/DRA1.tif',-1)

    dim1,dim2 = mask.shape

    Snw_dra = np.zeros(shape=(dim1, dim2,num_image_dra))
    counter = 0 

    for i in image_sequence_dra:
        I_dra=cv2.imread('Data_processed/Drainage_processed/DRA'+str(i)+'.tif',-1)
        Snw_dra[:,:,counter]=np.log(Iw)-np.log(I_dra)
        counter=counter+1

    time_dra = []

    for i in image_sequence_dra:
        path_file='Drainage/'+'DRATS'+str(i)+".fit"
        timestamp=os.path.getmtime(path_file)
        datestamp=datetime.fromtimestamp(timestamp)
        time_dra.append(datestamp)
        time_dra = sorted(time_dra)
    
    return dim1, image_sequence_dra, Iw, Snw_dra, time_dra

def constructor_redistribution(start_image_red,end_image_red,dim1,Iw,steps_divider=10):
    interval_red = (end_image_red-start_image_red)//steps_divider

    image_sequence_red = np.arange(start=start_image_red,stop=end_image_red,step=interval_red,dtype=int)
    num_image_red = len(image_sequence_red)

    snw_red = np.zeros(shape=(dim1, dim1,num_image_red))

    # Redistribution
    counter = 0 
    for i in image_sequence_red:
        I_red=cv2.imread("Data_processed/Redistribution_processed/RED"+str(i)+'.tif',-1)
        snw_red[:,:,counter]=np.log(Iw)-np.log(I_red)
        counter=counter+1

    return snw_red 

def compute_redistribution_sat(snw_red,alpha_pixel):
    snw_field_red = snw_red*alpha_pixel
    return snw_field_red

## Alpha calibration
def alpha_calibration(dim1,time_dra,Snw_dra,inj_rate,thick,poro,pixel_dim,por_vol):

    # Estimated Vnw
    time_elapsed=[(i-time_dra[0]).total_seconds()/60 for i in time_dra]
    Vnw = [inj_rate * i for i in time_elapsed]

    X = np.squeeze(np.sum(Snw_dra,(0,1))).reshape(-1,1)
    y = [i/(58.42**2*thick*poro) for i in Vnw]

    ## Fit linear model to data points (scikit-learn)

    mdl = LinearRegression(fit_intercept = False).fit(X,y)
    alpha_matched = mdl.coef_
    alpha_pixel = alpha_matched*dim1**2

    # Compute saturation field - drainage
    Snw_field_dra = Snw_dra*alpha_pixel

    # Compute Vnw through alpha calibration
    Vnw_computed = np.sum(Snw_field_dra*(pixel_dim**2)*thick*poro,axis=(0,1))

    # Compute Snw using Vnw computed
    eff_snw = Vnw_computed/por_vol *100

    return X , y , alpha_matched, alpha_pixel, Vnw, Vnw_computed,Snw_field_dra, eff_snw

## Spatial data analysis 
def height_snw(snw_field_dra,h,seq,por_vol,pixel_dim,thick,poro):
    
    dim1,dim2,dim3 = snw_field_dra.shape
    interval=h/pixel_dim
    dict_snw_2d={}
    vnw=snw_field_dra*(pixel_dim**2)*thick*poro
    for k in range(0,len(seq),10):

        arr=[]
        dist = []
        eff_snw = []
        
        for i in range(1,int(42/pixel_dim),int(interval)):
            
            ones=np.ones(shape=(dim1,dim2))
            ones[:-i]=0
            
            vnw1=vnw[:,:,k]*ones
            snw=(np.sum(vnw1,axis=(0,1))/por_vol)*100
            
            dist.append(i*pixel_dim)
            eff_snw.append(snw)
        arr= list(zip(dist,eff_snw))
        dict_snw_2d[k]= arr
        
    return dict_snw_2d

def initial_residual_curves(snw_field_dra,snw_field_red,title):
    x = snw_field_dra[:,:,-1].flatten()
    y = snw_field_red[:,:,-1].flatten()

    fig, ax = plt.subplots(figsize=(3.54,3.54),dpi=600)
    ax.scatter(x,y,color='green',linewidths=0.1,alpha=0.3,s=1)
    ax.axline([0,0],[1,1],linestyle='--',color='black',linewidth=1)
    fig.suptitle(title)
    ax.set_xlabel('Post Drainage Saturation')
    ax.set_ylabel('Post Redistribution Saturation')
    ax.set_xlim((0,1))
    ax.set_ylim((0,1))
    return plt.show()

## Plotting and video generation
def alpha_calibration_plots(X , y , alpha_matched, alpha_pixel, Vnw, Vnw_computed):
    ## Snw vs LnIw - LnI
    fig, ax = plt.subplots(figsize=(3.54,3.54),dpi=600)
    ax.scatter(X,y,color='blue',marker='x',linewidths=0.5,alpha=0.5,label='Data')
    ax.plot(X,alpha_matched*X,color='red',linestyle='dashdot',label='Linear model')
    ax.set_title('Alpha calibration')
    ax.set_xlabel('ln(Iw)-ln(I)')
    ax.set_ylabel('Snw')
    ax.text(np.max(X)+10000,np.mean(y)+0.005,'Alpha domain: '+str(np.round(alpha_matched,10)))
    ax.text(np.max(X)+10000,np.mean(y),'Alpha pixel: '+str(np.round(alpha_pixel,4)))
    ax.set_xlim(0,np.max(X))
    ax.set_ylim(0,np.max(y))
    ax.legend(loc='upper left')
    plt.ticklabel_format(axis='x',style='scientific',scilimits=(4,4))

    ## Vnw computed vs theoretical
    fig, ax = plt.subplots(figsize=(3.54,3.54),dpi=600)
    ax.scatter(Vnw,Vnw_computed,marker='o',alpha=0.5,linewidths=0.5,label='Data')
    ax.axline([0,0],[1,1],linestyle='--',color='black',linewidth=1)
    ax.set_title('Theoretical NWP volume vs Computed NWP volume')
    ax.set_ylabel('Pixel-tailed Vnw (ml)')
    ax.set_xlabel('Theoretical Vnw (ml)')
    ax.set_xlim(0,np.max(Vnw))
    ax.set_ylim(0,np.max(Vnw_computed))
    ax.legend()

    ## Discrepancy Vnw
    fig, ax = plt.subplots(figsize=(3.54,3.54),dpi=600)

    ax.scatter(np.arange(1,len(Vnw_computed)+1,1),Vnw_computed-Vnw,marker='o',alpha=0.5,linewidths=0.5,color='green')
    ax.set_title('Discrepancy in NWP Volume')
    ax.set_ylabel('Vnw (ml)')

    return plt.show()

### Saturation field
def plot_map(Sof,Iw,title):
    
    fig,ax = plt.subplots(figsize=(3.54,3.54),dpi=600)
    
    ax.imshow(Iw,cmap='gray')
    Sof_field=np.ma.masked_where(Sof<0.02,Sof)
    max_val=np.round(np.max(Sof_field),2)
    Sof_field_img=ax.imshow(Sof_field,cmap='viridis',interpolation='nearest',vmin=0,vmax=max_val)
    
    kwargs = {'format': '%.2f'}
    plt.colorbar(Sof_field_img,ax=ax,orientation='horizontal',shrink=0.6,ticks=[0,float(max_val)],**kwargs).set_label(label='Snw',size=7,weight='bold')
    
    plt.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,
        left=False,        # ticks along the top edge are off
        labelbottom=False,
        labelleft=False) # labels along the bottom edge are off
    
    plt.title(title,pad=10)

    return plt.show()

def height_eff_snw_plot(d,time):
    
    fig,ax = plt.subplots(figsize=(3.54,3.54),dpi=600)
    for key, value in d.items():
        y,x=zip(*value)
        ax.scatter(x,y,marker='x',s=5,label=key)
        ax.plot(x,y,linestyle='--',linewidth=0.5)
    plt.ylabel('Height (cm)')
    plt.xlabel('Effective Snw (%)')
    handles, labels = ax.get_legend_handles_labels()
    labels = ['t=0','t1','t2','t3','t4','t= '+ time]
    ax.legend(handles,labels,loc='upper center', bbox_to_anchor=(0.5, -0.17),
          fancybox=True, shadow=True,ncol=6,fontsize='x-small')
    return plt.show()

def video_generator (imgs,tiff_file_drainage,filename):
    frames = []
    fig = plt.figure()
    dim1,dim2,dim3 = imgs.shape

    img=plt.imread(tiff_file_drainage)
    img=plt.imshow(img,cmap='gray')

    Sof_field=np.ma.masked_where(imgs<0.05,imgs)

    for i in range(dim3):
        frames.append([plt.imshow(Sof_field[:,:,i],cmap='viridis',interpolation='nearest',vmin=0.1,vmax=1,animated=True)])

    ani = animation.ArtistAnimation(fig,frames,interval=3000,blit=True,repeat_delay=5000)

    plt.colorbar(orientation='vertical',shrink=0.8)
    
    f = r"Data_processed/"+filename
    writergif = animation.PillowWriter(fps=5) 
    ani.save(f, writer=writergif)

    return 
