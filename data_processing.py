# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 14:32:50 2022

@author: Jose
"""

## Import libraries

import numpy as np
import scipy.io
import cv2
import os
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
from sklearn.linear_model import LinearRegression

# Inputs

os.chdir('C:/Users/Jose/Box/2022-2023 GRA/Sand Tank Experiments/2022-4-16-P337-P170-ripples2/Processed-Data/')

pixel_dim = 0.0234 # cm
poro = 0.383
thick = 1.8
inj_rate = 0.185

# Drainage images

start_image_dra = 1 # NWP first seen in tank
end_image_dra =111 # domain breaktrough
interval_dra = 2

# Redistribution images 

start_image_red = 1 # NWP first seen in tank
end_image_red = 94 # domain breaktrough
interval_red = 10

mask_file=scipy.io.loadmat('BW.mat')
mask=mask_file.get('BW')

tiff_file_wet='Drainage/DRA1.tif'

# Whole domain saturation
image_sequence_dra = np.arange(start=start_image_dra,stop=end_image_dra+1,step=interval_dra,dtype=int)
num_image_dra = len(image_sequence_dra)

image_sequence_red = np.arange(start=start_image_red,stop=end_image_red+1,step=interval_red,dtype=int)
num_image_red = len(image_sequence_red)

dim1, dim2 = mask.shape

Snw_dra = np.zeros(shape=(dim1, dim2,num_image_dra))
Snw_red = np.zeros(shape=(dim1, dim2,num_image_red))

Iw = cv2.imread(tiff_file_wet,-1)


# Drainage
counter = 0 
for i in image_sequence_dra:
    I_dra=cv2.imread("Drainage/DRA"+str(i)+'.tif',-1)
    Snw_dra[:,:,counter]=np.log(Iw)-np.log(I_dra)
    counter=counter+1
    
# Redistribution
counter = 0 
for i in image_sequence_red:
    I_red=cv2.imread("Redistribution/IMB"+str(i)+'.tif',-1)
    Snw_red[:,:,counter]=np.log(Iw)-np.log(I_red)
    counter=counter+1
    
time_DRA = []

for i in image_sequence_dra:
    path_file='C:/Users/Jose/Box/2022-2023 GRA/Sand Tank Experiments/2022-4-16-P337-P170-ripples2/Drainage/'+'DRATS'+str(i)+".fit"
    timestamp=os.path.getmtime(path_file)
    datestamp=datetime.fromtimestamp(timestamp)
    time_DRA.append(datestamp)
    time_DRA = sorted(time_DRA)
    
time_elapsed=[(i-time_DRA[0]).total_seconds()/60 for i in time_DRA]

Vnw = [inj_rate * i for i in time_elapsed]

X = np.squeeze(np.sum(Snw_dra,(0,1))).reshape(-1,1)

y = [i/(58.42**2*thick*poro) for i in Vnw]

## Fit linear model to data points (scikit-learn)

mdl = LinearRegression(fit_intercept = False).fit(X,y)
alpha_matched = mdl.coef_
alpha_pixel = alpha_matched*dim1**2

# Compute saturation field - drainage
Snw_field_dra = Snw_dra*alpha_pixel

# Compute saturation field - redistribution
Snw_field_red = Snw_red*alpha_pixel

threshold = 0 

Vnw_computed = np.sum(Snw_field_dra*(pixel_dim**2)*thick*poro,axis=(0,1))

# Plots 
## Snw vs LnIw - LnI
fig, ax = plt.subplots(figsize=(3.54,3.54),dpi=600)

ax.scatter(X,y,color='blue',marker='x',linewidths=0.5,alpha=0.5,label='Data')
ax.plot(X,alpha_matched*X,color='red',linestyle='dashdot',label='Linear model')
ax.set_title('Alpha calibration')
ax.set_xlabel('ln(Iw)-ln(I)')
ax.set_ylabel('Snw')
ax.text(np.max(X)+10000,np.mean(y)+0.0005,'Alpha domain: '+str(np.round(alpha_matched,10)))
ax.text(np.max(X)+10000,np.mean(y),'Alpha pixel: '+str(np.round(alpha_pixel,4)))
ax.set_xlim(0,np.max(X)+2)
ax.set_ylim(0,np.max(y))
ax.legend()
plt.ticklabel_format(axis='x',style='scientific',scilimits=(4,4))
#plt.tight_layout()
plt.show()

## Vnw computed vs Vnw theoretical
fig, ax = plt.subplots(figsize=(3.54,3.54),dpi=600)

ax.scatter(Vnw,Vnw_computed,marker='o',alpha=0.5,linewidths=0.5,label='Data')
ax.axline([0,0],[1,1],linestyle='--',color='black',linewidth=1)
ax.set_title('Theoretical NWP volume vs Computed NWP volume')
ax.set_ylabel('Pixel-tailed Vnw (ml)')
ax.set_xlabel('Theoretical Vnw (ml)')
ax.set_xlim(0,np.max(Vnw))
ax.set_ylim(0,np.max(Vnw_computed))
ax.legend()
#plt.tight_layout()
plt.show()

## Discrepancy Vnw
fig, ax = plt.subplots(figsize=(3.54,3.54),dpi=600)

ax.scatter(np.arange(1,len(Vnw_computed)+1,1),Vnw_computed-Vnw,marker='o',alpha=0.5,linewidths=0.5,color='green')
ax.set_title('Discrepancy in NWP Volume')
ax.set_ylabel('Vnw (ml)')
#plt.tight_layout()
plt.show()

## Saturation field 
def plot_map(Sof,tiff_file_drainage,title):
    
    img=plt.imread(tiff_file_drainage)
    
    fig,ax = plt.subplots(figsize=(3.54,3.54),dpi=600)
    
    img=ax.imshow(img,cmap='gray')
    Sof_field=np.ma.masked_where(Sof<0.05,Sof)
    Sof_field_img=ax.imshow(Sof_field,cmap='viridis',interpolation='nearest',vmin=0.1,vmax=1)
    max_snw=Sof_field.max()
    
    plt.text(100,300,'Maximum Snw: '+str(np.round(max_snw,3)),
             bbox=dict(facecolor='blue', alpha=0.3),
             fontsize=5)
    
    plt.colorbar(Sof_field_img,ax=ax,orientation='vertical',shrink=0.8)
    
    plt.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,
        left=False,        # ticks along the top edge are off
        labelbottom=False,
        labelleft=False) # labels along the bottom edge are off
    
    plt.title(title,pad=10)
 #   plt.tight_layout()

    return plt.show()

def video_generator (imgs,tiff_file_drainage):
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
    
    f = r"C:/Users/Jose/Box/2022-2023 GRA/Sand Tank Experiments/2022-4-16-P337-P170-ripples2/Processed-Data/animation_ripples_337_170.gif" 
    writergif = animation.PillowWriter(fps=5) 
    ani.save(f, writer=writergif)

    return 




plot_map(Snw_field_dra[:,:,-1],tiff_file_wet,'Drainage')

plot_map(Snw_field_red[:,:,1],tiff_file_wet,'End of Drainage')

plot_map(Snw_field_red[:,:,-1],tiff_file_wet,'End of Redistribution')

video_generator (Snw_field_dra,tiff_file_wet)
