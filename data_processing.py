# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 14:32:50 2022

@author: Jose
"""

## Import libraries
import functions as ft
import os 

# Inputs
os.chdir('C:/Users/josee/Box/2022-2023 GRA/Sand Tank Experiments/2022_Ubillus_Experiments/01-25-2023_Ripples_337_080')

pixel_dim = 0.0253 # cm
poro = 0.26
thick = 1.8
inj_rate = 0.161
por_vol = 1035
h = 1
## Drainage inputs
start_image_dra =6 # NWP first seen in tank
end_image_dra = 2021 # domain breaktrough
## Redistribution inputs
start_image_red = 1
end_image_red = 91

dim1, image_sequence_dra, Iw, Snw_dra, time_dra = ft.constructor_drainage(start_image_dra,end_image_dra)
snw_red = ft.constructor_redistribution(start_image_red,end_image_red,dim1,Iw)
X, y, alpha_matched, alpha_pixel, Vnw, Vnw_computed,Snw_field_dra,eff_snw = ft.alpha_calibration(dim1,time_dra,Snw_dra,inj_rate,thick,poro,pixel_dim,por_vol)
snw_field_red = ft.compute_redistribution_sat(snw_red,alpha_pixel)
dict_snw_2d = ft.height_snw(Snw_field_dra,h,image_sequence_dra,por_vol,pixel_dim,thick,poro)

ft.alpha_calibration_plots(X, y, alpha_matched, alpha_pixel, Vnw, Vnw_computed)
ft.plot_map(Snw_field_dra[:,:,-1],Iw,'Drainage')
ft.height_eff_snw_plot(dict_snw_2d,time='1067 min')
ft.initial_residual_curves(Snw_field_dra,snw_field_red,'Exp. G')

#plot_map(Snw_field_dra[:,:,num_image_dra-1],tiff_file_wet,'Drainage')

#plot_map(Snw_field_red[:,:,1],tiff_file_wet,'End of Drainage')

#plot_map(Snw_field_red[:,:,num_image_red-1],tiff_file_wet,'End of Redistribution')

#video_generator (Snw_field_dra,tiff_file_wet)