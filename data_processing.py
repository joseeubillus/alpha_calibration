# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 14:32:50 2022

@author: Jose
"""

## Import libraries
import functions as ft
import plots as pt
import os 

# Inputs

dict_experiments = {'E':['2022-4-2_Ripples_337_230',14,55,568,1,115,0.0226,0.403,0.2,1511,'29 min'],
                    'F':['2022-4-16_Ripples_337_170',1,107,477,1,94,0.0234,0.383,0.185,1438,'56 min'],
                    'G':['01-05-2023_Ripples_337_140',1,683,2705,1,1463,0.0256,0.407,0.15,1570.57,'360 min'],
                    'H':['02-03-2023_Ripples_337_120',1,693,1865,1,91,0.0252,0.2785,0.128,1065,'350 min'],
                    'I':['01-25-2023_Ripples_337_080',6,553,2477,1,91,0.0253,0.2579,0.161,1035,'292 min']} 
#[path,start_drainage,domain_drainage,start_red,end_red,pixel_dim,poro,inj_rate,por_vol,time]

thick = 1.8
h = 1
kernel_size = 12

path_dict = dict_experiments['H'][0]
## Drainage inputs
start_image_dra = dict_experiments['H'][1]# NWP first seen in tank
domain_break = dict_experiments['H'][2] # domain breaktrough
end_image_dra = dict_experiments['H'][3]
## Redistribution inputs
start_image_red = dict_experiments['H'][4]
end_image_red = dict_experiments['H'][5]

pixel_dim = dict_experiments['H'][6]
poro = dict_experiments['H'][7]
inj_rate = dict_experiments['H'][8]
por_vol = dict_experiments['H'][9]
time = dict_experiments['H'][10]

os.chdir('C:/Users/josee/Box/2022-2023 GRA/Sand Tank Experiments/2022_Ubillus_Experiments/'+path_dict)


# Constructors
dim1, image_sequence_break, image_sequence_dra, Iw, Snw_break, time_break, Snw_dra, time_dra = ft.constructor_drainage(start_image_dra,domain_break,end_image_dra)

snw_red, image_sequence_red = ft.constructor_redistribution(start_image_red,end_image_red,dim1,Iw,steps_divider=50)

# Alpha calibration & saturation field
X, y , alpha_matched, alpha_pixel, Vnw = ft.alpha_calibration(dim1,time_break,Snw_break,inj_rate,thick,poro)
Snw_field_break, vnw_computed_break, eff_snw_break, Snw_field_dra, vnw_computed_dra, eff_snw_dra = ft.compute_drainage_sat(Snw_break, Snw_dra,alpha_pixel,pixel_dim,thick,poro,por_vol)
snw_field_red,eff_snw_red,vnw_computed_red = ft.compute_redistribution_sat(snw_red,alpha_pixel,pixel_dim,thick,poro,por_vol)


pt.alpha_calibration_plots(X, y, alpha_matched, alpha_pixel, Vnw, vnw_computed_break)
pt.plot_map('breakthrough',Snw_field_break[:,:,-1],Iw,eff_snw_break[-1],time)
