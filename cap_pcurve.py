import functions as ft
import numpy as np
import matplotlib.pyplot as plt
import os 

# Define the data dictionary
dict_experiments = {'E':['2022-4-2_Ripples_337_230',14,55,568,1,115,0.0226,0.403,0.2,1511,'29 min'],
                    'F':['2022-4-16_Ripples_337_170',1,107,477,1,94,0.0234,0.383,0.185,1438,'56 min'],
                    'G':['01-05-2023_Ripples_337_140',1,683,2705,1,1463,0.0256,0.407,0.15,1570.57,'360 min'],
                    'H':['02-03-2023_Ripples_337_120',1,693,1865,1,91,0.0252,0.2785,0.128,1065,'350 min'],
                    'I':['01-25-2023_Ripples_337_080',6,553,2477,1,91,0.0253,0.2579,0.161,1035,'292 min'],
                    'J':['03-02-2023_Climbing_Ripples_337_140',18,1864,2213,1,6,0.0244,0.30,0.145,1122,'984 min'],
                    'K':['03-31-2023_Climbing_Ripples_337_140_washed',33,1074,2390,1,66,0.025,0.312,0.173,1182,'555 min'],
                    'L':['05-08-2023_Climbing_Ripples_337_140_2',20,1069,2427,1,32,0.0252,0.28,0.18,1080,'560 min']}


thick = 1.8
h = 1

path_dict = dict_experiments['G'][0]
## Drainage inputs
start_image_dra = dict_experiments['G'][1]# NWP first seen in tank
domain_break = dict_experiments['G'][2] # domain breaktrough
end_image_dra = dict_experiments['G'][3]

os.chdir('C:/Users/ubillusj/Box/2022-2023 GRA/Sand Tank Experiments/2022_Ubillus_Experiments/'+path_dict)

# Constructors
dim1, image_sequence_break, image_sequence_dra, Iw, Snw_break, time_break, Snw_dra, time_dra = ft.constructor_drainage(start_image_dra,domain_break,end_image_dra)

# Upload the pressure data
num, date, pout,pin = np.loadtxt('Ripples2.csv',delimiter=',',skiprows=2,usecols=(0,1,2,3),unpack=True)
