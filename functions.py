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
def constructor_drainage(start_image_dra,domain_break_image,end_image_dra,steps_divider=50):

    mask_file=scipy.io.loadmat('Data_processed/BW.mat')
    mask=mask_file.get('BW')

    dim1,dim2 = mask.shape  
    
    interval_dra = (domain_break_image-start_image_dra)//steps_divider   
    image_sequence_break = np.arange(start=start_image_dra,stop=domain_break_image+1,step=interval_dra,dtype=int)
    image_sequence_dra = np.arange(start=domain_break_image,stop=end_image_dra+1,step=steps_divider,dtype=int)
    num_image_break = len(image_sequence_break)
    num_image_dra = len(image_sequence_dra)

    Iw = cv2.imread('Data_processed/Drainage_processed/DRA1.tif',-1)

    Snw_break = np.zeros(shape=(dim1, dim2,num_image_break))
    Snw_dra = np.zeros(shape=(dim1, dim2,num_image_dra))
    time_break = []
    time_dra = []

    counter = 0 

    for i  in image_sequence_break:
        I_break=cv2.imread('Data_processed/Drainage_processed/DRA'+str(i)+'.tif',-1)
        Snw_break[:,:,counter]=np.log(Iw)-np.log(I_break)
        path_file_break='Drainage/'+'DRATS'+str(i)+".fit"
        timestamp_break=os.path.getmtime(path_file_break)
        datestamp_break=datetime.fromtimestamp(timestamp_break)
        time_break.append(datestamp_break)
        time_break = sorted(time_break)
        counter = counter +1

    counter = 0
    for j in image_sequence_dra:
        I_dra=cv2.imread('Data_processed/Drainage_processed/DRA'+str(j)+'.tif',-1)
        Snw_dra[:,:,counter]=np.log(Iw)-np.log(I_dra)
        path_file='Drainage/'+'DRATS'+str(j)+".fit"
        timestamp=os.path.getmtime(path_file)
        datestamp=datetime.fromtimestamp(timestamp)
        time_dra.append(datestamp)
        time_dra = sorted(time_dra)
        counter = counter +1 
    
    return dim1, image_sequence_break, image_sequence_dra, Iw, Snw_break, time_break, Snw_dra, time_dra

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

    return snw_red, image_sequence_red

## Alpha calibration
def alpha_calibration(dim1,time_break,Snw_break,inj_rate,thick,poro):

    # Estimated Vnw
    time_elapsed=[(i-time_break[0]).total_seconds()/60 for i in time_break]
    Vnw = [inj_rate * i for i in time_elapsed]

    X = np.squeeze(np.sum(Snw_break,(0,1))).reshape(-1,1)
    y = [i/(58.42**2*thick*poro) for i in Vnw]

    ## Fit linear model to data points (scikit-learn)

    mdl = LinearRegression(fit_intercept = False).fit(X,y)
    alpha_matched = mdl.coef_
    alpha_pixel = alpha_matched*dim1**2

    return X, y , alpha_matched, alpha_pixel, Vnw

def compute_drainage_sat(Snw_break, Snw_dra,alpha_pixel,pixel_dim,thick,poro,por_vol):
    # Compute saturation field - drainage
    Snw_field_break = Snw_break*alpha_pixel
    Snw_field_dra = Snw_dra*alpha_pixel

    # Compute Vnw through alpha calibration
    vnw_computed_break = np.sum(Snw_field_break*(pixel_dim**2)*thick*poro,axis=(0,1))
    vnw_computed_dra = np.sum(Snw_field_dra*(pixel_dim**2)*thick*poro,axis=(0,1))

    # Compute Snw using Vnw computed
    eff_snw_break = vnw_computed_break/por_vol
    eff_snw_dra = vnw_computed_dra/por_vol

    return Snw_field_break, vnw_computed_break, eff_snw_break, Snw_field_dra, vnw_computed_dra, eff_snw_dra

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
def height_snw(snw_field_dra,middle,h,seq,pixel_dim,kernel_size):
    
    interval=h/pixel_dim
    dict_snw_2d={}

    for k in [0,middle,(len(seq)-1)]:

        arr=[]
        dist = []
        eff_snw = []
        
        for i in range(1,int(42/pixel_dim),int(interval)):
            
            snw1=snw_field_dra[-i,:,k]
            snw_cm=np.round(np.mean(snw1),2)
            dist.append(i*pixel_dim)
            eff_snw.append(snw_cm)

        kernel = np.ones(kernel_size)/kernel_size
        convolve_eff_snw = np.convolve(eff_snw,kernel,'same')
        arr= list(zip(dist,convolve_eff_snw))
        dict_snw_2d[k]= arr
        
    return dict_snw_2d

def spatial_moment(snw_field_break,snw_field_dra,snw_field_red,img_seq_break,img_seq_dra,img_seq_red,poro,thick,pixel_dim,dim1):

    time_break = []
    time_dra = []
    time_red = []

    m00_break = []
    m00_dra = []
    m00_red = []

    m10_break = []
    m10_dra = []
    m10_red = []

    m01_break = []
    m01_dra = []
    m01_red = []

    pixel_vol = pixel_dim**2 * thick
    arrays = [np.arange(1,dim1+1,1) for _ in range(dim1)]
    position_x = np.stack(arrays,axis=0) *pixel_dim
    position_z = np.flipud(np.stack(arrays,axis=1)) *pixel_dim

    for i in range(0,int(len(img_seq_break)),1):

        temp = np.sum(pixel_vol * poro * snw_field_break[:,:,i])
        temp10 = np.sum(pixel_vol * poro * snw_field_break[:,:,i] * position_x)  
        temp01 = np.sum(pixel_vol * poro * snw_field_break[:,:,i] * position_z)  
        time_break.append(img_seq_break[i]*0.5)

        m00_break.append(temp)
        m10_break.append(temp10)
        m01_break.append(temp01)

    for i in range(0,int(len(img_seq_dra)),1):
                
        temp = np.sum(pixel_vol * poro * snw_field_dra[:,:,i]) 
        temp10 = np.sum(pixel_vol * poro * snw_field_dra[:,:,i] * position_x)  
        temp01 = np.sum(pixel_vol * poro * snw_field_dra[:,:,i] * position_z)              
        time_dra.append(img_seq_dra[i]*0.5)

        m00_dra.append(temp)
        m10_dra.append(temp10)
        m01_dra.append(temp01)

    for i in range(0,int(len(img_seq_red)),1):
                
        temp = np.sum(pixel_vol * poro * snw_field_red[:,:,i])              
        time_red.append(time_dra[-1]+img_seq_red[i]*0.5)
        temp10 = np.sum(pixel_vol * poro * snw_field_red[:,:,i] * position_x)  
        temp01 = np.sum(pixel_vol * poro * snw_field_red[:,:,i] * position_z)

        m00_red.append(temp)
        m10_red.append(temp10)
        m01_red.append(temp01)
    
        x_brk = np.divide(m10_break,m00_break)
        z_brk = np.divide(m01_break,m00_break)
    
        x_dra = np.divide(m10_dra,m00_dra)
        z_dra = np.divide(m01_dra,m00_dra)
    
        x_red = np.divide(m10_red,m00_red)
        z_red = np.divide(m01_red,m00_red)

    return time_break, time_dra, time_red, m00_break, m00_dra, m00_red, x_brk, z_brk, x_dra, z_dra, x_red, z_red










