# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 14:32:50 2022

@author: Jose
"""

## Import libraries
import functions as ft
import os 

# Inputs
os.chdir('C:/Users/josee/Box/2022-2023 GRA/Sand Tank Experiments/2022_Ubillus_Experiments/02-03-2023_Ripples_337_120/')

pixel_dim = 0.0252 # cm
poro = 0.28
thick = 1.8
inj_rate = 0.128
por_vol = 1066
h = 1
## Drainage inputs
start_image_dra =1 # NWP first seen in tank
end_image_dra = 693 # domain breaktrough

dim1, image_sequence_dra, Iw, Snw_dra, time_dra = ft.constructor_drainage(start_image_dra,end_image_dra)
X, y, alpha_matched, alpha_pixel, Vnw, Vnw_computed,Snw_field_dra,eff_snw = ft.alpha_calibration(dim1,time_dra,Snw_dra,inj_rate,thick,poro,pixel_dim,por_vol)
dict_snw_2d = ft.height_snw(Snw_field_dra,h,image_sequence_dra,por_vol,pixel_dim,thick,poro)

ft.alpha_calibration_plots(X, y, alpha_matched, alpha_pixel, Vnw, Vnw_computed)
ft.plot_map(Snw_field_dra[:,:,-1],Iw,'Drainage')
ft.height_eff_snw_plot(dict_snw_2d)
'''
# Redistribution images 

start_image_red = 1 # NWP first seen in tank
end_image_red = 91 # domain breaktrough
interval_red = (end_image_dra-start_image_dra)//10
'''

'''
image_sequence_red = np.arange(start=start_image_red,stop=end_image_red,step=interval_red,dtype=int)
num_image_red = len(image_sequence_red)
'''
#Snw_red = np.zeros(shape=(dim1, dim2,num_image_red))


'''
# Redistribution
counter = 0 
for i in image_sequence_red:
    I_red=cv2.imread("Redistribution_processed/RED"+str(i)+'.tif',-1)
    Snw_red[:,:,counter]=np.log(Iw)-np.log(I_red)
    counter=counter+1
'''

# Compute saturation field - redistribution
#Snw_field_red = Snw_red*alpha_pixel
 
# Plots 


## Saturation field 

#plot_map(Snw_field_dra[:,:,num_image_dra-1],tiff_file_wet,'Drainage')

#plot_map(Snw_field_red[:,:,1],tiff_file_wet,'End of Drainage')

#plot_map(Snw_field_red[:,:,num_image_red-1],tiff_file_wet,'End of Redistribution')

#video_generator (Snw_field_dra,tiff_file_wet)

