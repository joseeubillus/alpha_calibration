import streamlit as st
import numpy as np
import pandas as pd
import scipy.io
import cv2
import os
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from PIL import Image

header= Image.open('C:/Users/Jose/Desktop/alpha_calibration/GCCC_header.png')
#Backend initial values

thick = 1.8 #constant variable
inj_rate = 0.2 #constant variable

st.image(header,use_column_width=True,caption='Gulf Coast Carbon Center')
st.title('CO2 SATURATION DATA PROCESSING - ALPHA CALIBRATION')

st.text('Developed by: Jose Ubillus')

st.text('''Draft - Work in progress
Steps are shown in the data processing procedure and functions for calculations are 
in the file alpha calibration.py
	''')

st.header('Experiment directory path')

st.text('''Remember to use / instead of \ when adding the files path\
    
    Note: add / at the end of the raw path
    
    ''')

raw_path=st.text_input('Insert raw files path directory')
processed_path=st.text_input('Insert drainage processed files path directory')
red_path=st.text_input('Insert redistribution files path')
wet_path=st.text_input('Insert wet image path directory')
a=st.number_input('Insert a or b value given by ImageJ')

process_start=st.checkbox('Check once you have enter all the information above')

if process_start==True:
        
    st.header('Step 1: Porosity and pixel dimension estimation')

    st.subheader('Pixel dimension estimation')
    pixel_dim=58.42/a
    st.text(f'Pixel dimension is {round(pixel_dim,3)}')

    st.subheader('Porosity estimation')

    h1=st.number_input('Insert first filling image pixel height')
    w1=st.number_input('Insert first filling image wetting volume ')

    h2=st.number_input('Insert second filling image pixel height')
    w2=st.number_input('Insert second filling image wetting volume ')

    h3=st.number_input('Insert third filling image pixel height')
    w3=st.number_input('Insert third filling image wetting volume ')

    poro_check=st.checkbox('Check to see porosity estimation table and average porosity')
    if poro_check == True:
            
        por_df= pd.DataFrame({'Pixel':[h1,h2,h3],'WP Volume (ml)':[w1,w2,w3]})
        por_df['Height (cm)']=por_df['Pixel']*pixel_dim
        por_df['Height difference (cm)']=por_df['Height (cm)'].diff()
        por_df['WP Vol difference (ml)']=por_df['WP Volume (ml)'].diff()
        por_df['Bulk Volume']=thick*58.42*por_df['Height difference (cm)']
        por_df['Porosity']=por_df['WP Vol difference (ml)']/por_df['Bulk Volume']

        average_por = por_df['Porosity'].mean()

        st.dataframe(por_df)
        st.text(f'Porosity is estimated to be {round(average_por,3)}')

    st.header('Step 2: Alpha calibration')

    start_image=st.number_input('Insert image number of NWP first seen in tank')
    end_image=st.number_input('Insert image number of domain NWP breakthrough')
    end_image_red=st.number_input('Insert last image number in redistribution stage')
    
    alpha_check=st.checkbox('Check to run alpha calibration process')
    if alpha_check==True:
            
        # Backend alpha calibration
        os.chdir(raw_path)
        interval = (end_image-start_image)//50
        interval_red=(end_image-1)//10  
        image_sequence = np.arange(start=int(start_image),stop=int(end_image),step=interval,dtype=int)
        num_image = int(len(image_sequence))

        image_sequence_red = np.arange(start=1,stop=int(end_image_red),step=interval_red,dtype=int)
        num_image_red = int(len(image_sequence_red))

        dim1, dim2 = int(a)+1, int(a)+1

        Snw = np.zeros(shape=(dim1, dim2,num_image))
        Snw_red=np.zeros(shape=(dim1,dim2,num_image))


        Iw = cv2.imread(wet_path,-1)

        #Drainage
        counter = 0 
        for i in image_sequence:
            I=cv2.imread(processed_path+"/DRA"+str(i)+'.tif',-1)
            Snw[:,:,counter]=np.log(Iw)-np.log(I)
            counter=counter+1

        # Redistribution
        counter = 0 
        for i in image_sequence_red:
            I_red=cv2.imread(red_path+"/RED"+str(i)+'.tif',-1)
            Snw_red[:,:,counter]=np.log(Iw)-np.log(I_red)
            counter=counter+1

        time_DRA = []

        for i in image_sequence:
            path_file='Drainage/DRATS'+str(i)+".fit"
            timestamp=os.path.getmtime(path_file)
            datestamp=datetime.fromtimestamp(timestamp)
            time_DRA.append(datestamp)
            time_DRA = sorted(time_DRA)

        time_elapsed=[(i-time_DRA[0]).total_seconds()/60 for i in time_DRA]

        Vnw = [inj_rate * i for i in time_elapsed]

        X = np.squeeze(np.sum(Snw,(0,1))).reshape(-1,1)

        y = [i/(58.42**2*thick*average_por) for i in Vnw]

        ## Fit linear model to data points (scikit-learn)

        mdl = LinearRegression(fit_intercept = False).fit(X,y)
        alpha_matched = mdl.coef_
        alpha_pixel = alpha_matched*dim1**2

        # Compute saturation field - drainage
        Snw_field = Snw*alpha_pixel

        # Compute saturation field - redistribution
        Snw_field_red = Snw_red*alpha_pixel

        Vnw_computed = np.sum(Snw_field*(pixel_dim**2)*thick*average_por,axis=(0,1))

        def plot_map(Sof,tiff_file_drainage,title):
            
            threshold = 0.05
            img=plt.imread(tiff_file_drainage)
            
            fig,ax = plt.subplots(figsize=(3.54,3.54),dpi=600)
            
            img=ax.imshow(img,cmap='gray')
            Sof_field=np.ma.masked_where(Sof<threshold,Sof)
            Sof_field_img=ax.imshow(Sof_field,cmap='viridis',interpolation='nearest',vmin=0.1,vmax=1)
            max_snw=Sof_field.max()
            
            plt.text(100,300,'Maximum Snw: '+str(np.round(max_snw,3)),
                    bbox=dict(facecolor='blue', alpha=0.3),
                    fontsize=4)
            
            plt.colorbar(Sof_field_img,ax=ax,orientation='vertical',shrink=0.8)
            
            plt.tick_params(
                axis='both',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,
                left=False,        # ticks along the top edge are off
                labelbottom=False,
                labelleft=False) # labels along the bottom edge are off
            
            ax.set_title(title,fontdict={"fontsize":4})
            plt.tight_layout()

            return st.pyplot(fig)

st.header('Results')

results=st.checkbox('Check the box to show image processing and calibration results')

if results == True:

    fig1, ax = plt.subplots(figsize=(3.54,3.54),dpi=600)

    ax.scatter(X,y,color='blue',marker='x',linewidths=0.5,alpha=0.5,label='Data')
    ax.plot(X,alpha_matched*X,color='red',linestyle='dashdot',label='Linear model')
    ax.set_title('Alpha calibration')
    ax.set_xlabel('ln(Iw)-ln(I)')
    ax.set_ylabel('Snw')
    ax.text(np.max(X)+10000,np.mean(y)+0.005,'Alpha domain: '+str(np.round(alpha_matched,10)))
    ax.text(np.max(X)+10000,np.mean(y),'Alpha pixel: '+str(np.round(alpha_pixel,4)))
    ax.set_xlim(0,np.max(X))
    ax.set_ylim(0,np.max(y))
    ax.legend()
    plt.ticklabel_format(axis='x',style='scientific',scilimits=(4,4))
    st.pyplot(fig1)

    st.markdown("---")
    col1, col2 =st.columns(2)
    with col1:


        fig3, ax = plt.subplots(figsize=(3.54,3.54),dpi=600)

        ax.scatter(np.arange(1,len(Vnw_computed)+1,1),Vnw_computed-Vnw,marker='o',alpha=0.5,linewidths=0.5,color='green')
        ax.set_title('Discrepancy in NWP Volume')
        ax.set_ylabel('Vnw (ml)')
    
        st.pyplot(fig3)

    with col2:
        fig2, ax = plt.subplots(figsize=(3.54,3.54),dpi=600)

        ax.scatter(Vnw,Vnw_computed,marker='o',alpha=0.5,linewidths=0.5,label='Data')
        ax.axline([0,0],[1,1],linestyle='--',color='black',linewidth=1)
        ax.set_title('Theoretical vs Computed NWP volume')
        ax.set_ylabel('Pixel-tailed Vnw (ml)')
        ax.set_xlabel('Theoretical Vnw (ml)')
        ax.set_xlim(0,np.max(Vnw))
        ax.set_ylim(0,np.max(Vnw_computed))
        ax.legend()

        st.pyplot(fig2)

    st.markdown("---")
    plot_map(Snw_field[:,:,num_image-1],wet_path,' Breaktrough Drainage')
    st.markdown("---")
    plot_map(Snw_field_red[:,:,1],wet_path,'End of Drainage')
    st.markdown("---")
    plot_map(Snw_field_red[:,:,num_image_red-1],wet_path,'End of Redistribution')