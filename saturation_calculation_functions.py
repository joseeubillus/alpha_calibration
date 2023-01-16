# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 22:42:11 2022

@author: Jose
"""

# Import libraries 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import cv2
import scipy.io
import seaborn as sns
from sensitivity import SensitivityAnalyzer

#----------------------------------------

# Defining functions for saturation calculation

def read_image_create_mask(tiff_file_wet,tiff_file_dry,tiff_file_drainage):
    
    img_wet = cv2.imread(tiff_file_wet,-1).astype('uint16') # reading the image
    img_dry_register=cv2.imread(tiff_file_dry,-1).astype('uint16')
    img_drainage=cv2.imread(tiff_file_drainage,-1).astype('uint16')

    mask_file=scipy.io.loadmat('BW.mat')
    mask=mask_file.get('BW')
    

    return img_wet, img_dry_register,img_drainage, mask

def kvalue(img_wet,img_dry_register,naq,nair,ng,poro,thickness):
    
    
    #Calculations
    Tgw=((4*ng*naq))/((ng+naq)**2)
    
    Tga=((4*ng*nair))/((ng+nair)**2)
    
    IR = img_wet/img_dry_register
    
    TR=(Tgw/Tga)
    
    K=(np.log(IR))/(2*np.log(TR))
    
    TV=poro*thickness
    
    Do=TV/K
    
    return IR, Do, K
    
def SinglePoreOilSaturation(img_wet,img_dry_register,no,naq,A,poro,thickness,Do,K):
    
    Ioil_SP=np.zeros(img_wet.shape[:2],dtype='float64')
    Ioil=np.zeros(img_wet.shape[:2],dtype='float64')
    
    #Calculations
    Two=((4*no*naq))/((no+naq)**2)
    
    Ao=(2.303*A)/0.5 #%(2.303*A/T), where T is light path which is 1cm. June 14 it is 5 mm

    Ioil_SP=img_wet*(Two**2)*(np.exp(-Ao*Do))
    
    Ioil=img_wet*(Two**(2*K))*(np.exp(-Ao*Do*K))
    
    So=(np.log(img_wet)-np.log(Ioil_SP))/((np.log(img_wet))-np.log(Ioil))
    
    Spo=np.nanmean(So)
    
    return Ioil_SP, Ioil, So, Spo, Ao   

def SaturationCalculation(img_wet,img_drainage,mask,Ioil,Spo,pixel_dim,thickness,poro,no,naq):
    
    Sof=np.zeros(img_wet.shape[:2],dtype='float64') #filtered So to reduce noise
    
    #Calculations
    Two=((4*no*naq))/((no+naq)**2)
    
    Ioil=img_wet*(Two**(2*K))*(np.exp(-Ao*Do*K))
    
    So_sat=(np.log(img_wet)-np.log(img_drainage))/((np.log(img_wet))-np.log(Ioil))
    
    Sof=So_sat
    Sof=np.nan_to_num(Sof)
    Sof[Sof<Spo]=0
    

    
    NW_Vol=mask*Sof*(pixel_dim**2)*thickness*poro
    NW_Vol_Total=np.sum(NW_Vol)
    W_Vol=mask*(pixel_dim**2)*thickness*poro
    W_Vol_Total=np.sum(W_Vol)
    Sat=NW_Vol_Total/W_Vol_Total
    
    return So_sat, Sof, NW_Vol, W_Vol, NW_Vol_Total, W_Vol_Total, Sat

#-------------------------

# Defining the variables use for the saturation calculations

no=1.2
nair=1
ng=1.52 # %RI of glass beads from Potter Industries
naq=1.398 # %50:50 solution Glycerol-Water
A=1.474 # %40mg from June 14 measurements REMARK
poro=0.4
thickness=2.5
pixel_dim=0.0287

# Image processing and calculations

img_wet, img_dry_register, img_drainage, mask = read_image_create_mask('DRA1.tif', 'DRY_Register.tif','DRA3496.tif')
IR, Do, K= kvalue(img_wet, img_dry_register,naq,nair,ng,poro,thickness)
Ioil_SP, Ioil, So, Spo, Ao=SinglePoreOilSaturation(img_wet, img_dry_register, no, naq, A, poro, thickness, Do, K)
So_sat, Sof, NW_Vol, W_Vol, NW_Vol_Total, W_Vol_Total, Sat = SaturationCalculation(img_wet, img_drainage, mask, Ioil, Spo, pixel_dim, thickness, poro,no,naq)

#-------------------------

# Sensitivity Analysis of the impact of the variables in saturation and NW vol

no = [1.05,1.1,1.15,1.2,1.25,1.3,1.35]
naq = [1.3707,1.3774,1.38413,1.39089,1.39809]

res = []
df=[]
for i, nov in enumerate(no):
    for j, naqv in enumerate(naq):
        So_sat, Sof, NW_Vol, W_Vol, NW_Vol_Total, W_Vol_Total, Sat = SaturationCalculation(img_wet, img_drainage, mask, Ioil, Spo, pixel_dim, thickness, poro,nov,naqv)
        
        df.append(
            {
                'No':nov,
                'Naq':naqv,
                'Saturation':Sat,
                'NW Volume': NW_Vol_Total
                })
        
df=pd.DataFrame(df)




# Heatmap to see the influence of the variables on the response
pivot_saturation=pd.pivot_table(df,index='No',columns='Naq',values='Saturation')
pivot_NWV=pd.pivot_table(df,index='No',columns='Naq',values='NW Volume')


sat_heat=sns.heatmap(pivot_saturation,cmap='viridis',annot=True,fmt='.3f')
sat_heat.set_title("Saturation Heatmap varying No and Naq")
plt.show()

nw_heat=sns.heatmap(pivot_NWV,cmap='viridis',annot=True,fmt='.2f')
nw_heat.set_title("Non Wetting Volume Heatmap varying No and Naq")
plt.show()


























