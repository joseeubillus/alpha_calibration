import streamlit as st
import numpy as np
import scipy.io
import cv2
import os
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.title('CO2 SATURATION DATA PROCESSING - ALPHA CALIBRATION')

st.text('Developed by: Jose Ubillus')

st.text('''Draft - Work in progress
Steps are shown in the data processing procedure and functions for calculations are 
in the file alpha calibration.py
	''')

st.header('Experiment directory path')

raw_path=st.text_input('Insert raw files path directory')
processed_path=st.text_input('Insert processed files path directory')
wet_path=st.text_input('Insert wet image path directory')

#Backend initial values

thick = 1.8 #constant variable
inj_rate = 0.2 #constant variable

mask_file=scipy.io.loadmat('BW.mat')
mask=mask_file.get('BW')


st.header('Step 1: Porosity and pixel dimension estimation')

st.text('Porosity is estimated to be 0.4 overall')
a=st.number_input('Insert a or b value given by ImageJ')

st.header('Step 2: Alpha calibration')

start_image=st.number_input('Insert image number of NWP first seen in tank')
end_image=st.number_input('Insert image number of domain NWP breakthrough')

# Backend alpha calibration
image_sequence = np.arange(start=start_image,stop=end_image+interval,step=interval,dtype=int)
num_image = len(image_sequence)

dim1, dim2 = mask.shape

Snw = np.zeros(shape=(dim1, dim2,num_image))

Iw = cv2.imread(wet_path,-1)

counter = 0 
for i in image_sequence:
    I=cv2.imread(processed_path+"Drainage_processed/DRA"+str(i)+'.tif',-1)
    Snw[:,:,counter]=np.log(Iw)-np.log(I)
    counter=counter+1

time_DRA = []

for i in image_sequence:
    path_file=raw_path+'DRATS'+str(i)+".fit"
    timestamp=os.path.getmtime(path_file)
    datestamp=datetime.fromtimestamp(timestamp)
    time_DRA.append(datestamp)
    time_DRA = sorted(time_DRA)

time_elapsed=[(i-time_DRA[0]).total_seconds()/60 for i in time_DRA]

Vnw = [inj_rate * i for i in time_elapsed]

X = np.squeeze(np.sum(Snw,(0,1))).reshape(-1,1)

y = [i/(58.42**2*thick*poro) for i in Vnw]

## Fit linear model to data points (scikit-learn)

mdl = LinearRegression(fit_intercept = False).fit(X,y)
alpha_matched = mdl.coef_
alpha_pixel = alpha_matched*dim1**2

# Compute saturation field
Snw_field = Snw*alpha_pixel
threshold = 0 
Vnw_computed = np.sum(Snw_field*(pixel_dim**2)*thick*poro,axis=(0,1))

st.header('Results')

