'''
    This file contains all the functions used for data processing and calibration, 
    if any change is needed please refer to this file.

    Authors: Jose Ubillus and Hailun Ni 

'''
# Import require libraries
import numpy as np
import scipy.io
import cv2
import os
from datetime import datetime
from sklearn.linear_model import LinearRegression

## Constructors
def constructor_drainage(start_image_dra,end_image_dra,steps_divider=50):

    mask_file=scipy.io.loadmat('Data_processed/BW.mat')
    mask=mask_file.get('BW')

    interval_dra = (end_image_dra-start_image_dra)//steps_divider   
    image_sequence_dra = np.arange(start=start_image_dra,stop=end_image_dra+1,step=interval_dra,dtype=int)
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

## Alpha calibration
def alpha_calibration(dim1,time_dra,Snw_dra,inj_rate,thick,poro):

    # Estimated Vnw
    time_elapsed=[(i-time_dra[0]).total_seconds()/60 for i in time_dra]
    Vnw = [inj_rate * i for i in time_elapsed]

    X = np.squeeze(np.sum(Snw_dra,(0,1))).reshape(-1,1)
    y = [i/(58.42**2*thick*poro) for i in Vnw]

    ## Fit linear model to data points (scikit-learn)

    mdl = LinearRegression(fit_intercept = False).fit(X,y)
    alpha_matched = mdl.coef_
    alpha_pixel = alpha_matched*dim1**2

    return X, y , alpha_matched, alpha_pixel, Vnw

def compute_drainage_saturation(Snw_dra,alpha_pixel,pixel_dim,thick,poro,por_vol):
    # Compute saturation field - drainage
    Snw_field_dra = Snw_dra*alpha_pixel

    # Compute Vnw through alpha calibration
    vnw_computed_dra = np.sum(Snw_field_dra*(pixel_dim**2)*thick*poro,axis=(0,1))

    # Compute Snw using Vnw computed
    eff_snw_dra = vnw_computed_dra/por_vol

    return Snw_field_dra, vnw_computed_dra,eff_snw_dra

def compute_redistribution_sat(snw_red,alpha_pixel,pixel_dim,thick,poro,por_vol):

    snw_field_red = snw_red*alpha_pixel

    # Compute Vnw through alpha calibration
    vnw_computed_red = np.sum(snw_field_red*(pixel_dim**2)*thick*poro,axis=(0,1))

    # Compute Snw using Vnw computed
    eff_snw_red = vnw_computed_red/por_vol

    return snw_field_red,eff_snw_red,vnw_computed_red

def pixel_wise_saturation(snw):
    
    snw[snw<0]=0
    avg_px_sat = np.average(snw)

    return avg_px_sat

def trapping_efficiency(sat_red,sat_dra):

    trap_eff = sat_red/sat_dra

    return trap_eff
    
## Spatial data analysis 
def height_snw(snw_field_dra,h,seq,por_vol,pixel_dim,thick,poro,kernel_size):
    
    dim1,dim2,dim3 = snw_field_dra.shape
    interval=h/pixel_dim
    dict_snw_2d={}
    vnw=snw_field_dra*(pixel_dim**2)*thick*poro
    
    for k in range(0,len(seq),10):

        arr=[]
        dist = []
        eff_snw = []
        
        for i in range(1,int(42/pixel_dim),int(interval)):
            
            ones=np.zeros(shape=(dim1,dim2))
            ones[-i]=1
            
            vnw1=vnw[:,:,k]*ones
            snw=np.round(np.sum(vnw1,axis=(0,1)),2)/por_vol*100
            
            dist.append(i*pixel_dim)
            eff_snw.append(snw)

        kernel = np.ones(kernel_size)/kernel_size
        convolve_eff_snw = np.convolve(eff_snw,kernel,'same')
        arr= list(zip(dist,convolve_eff_snw))
        dict_snw_2d[k]= arr
        
    return dict_snw_2d



