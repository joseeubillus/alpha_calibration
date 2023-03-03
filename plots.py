'''
    This file contains all plots available for data processing: saturation field maps, trapping efficiency,
    pixel-wise trapping, height vs snw.

    Authors: Jose Ubillus and Hailun Ni
'''
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

## Plotting and video generation
def alpha_calibration_plots(X , y , alpha_matched, alpha_pixel, Vnw, Vnw_computed):
    ## Snw vs LnIw - LnI
    fig, ax = plt.subplots(figsize=(3.54,3.54),dpi=600)
    ax.scatter(X,y,color='blue',marker='x',linewidths=0.5,alpha=0.5,label='Data')
    ax.plot(X,alpha_matched*X,color='red',linestyle='dashdot',label='Linear model')
    
    fig.suptitle('Alpha calibration')
    ax.set_xlabel('ln(Iw)-ln(I)')
    ax.set_ylabel('Snw')
    ax.text(np.max(X)+10000,np.mean(y)+0.005,'Alpha domain: '+str(np.round(alpha_matched,10)))
    ax.text(np.max(X)+10000,np.mean(y),'Alpha pixel: '+str(np.round(alpha_pixel,4)))
    ax.set_xlim(0,np.max(X))
    ax.set_ylim(0,np.max(y))
    ax.legend(loc='upper left')
    plt.ticklabel_format(axis='x',style='scientific',scilimits=(4,4))

    ## Vnw computed vs theoretical
    fig, ax = plt.subplots(figsize=(3.54,3.54),dpi=600)
    ax.scatter(Vnw,Vnw_computed,marker='o',alpha=0.5,linewidths=0.5,label='Data')
    ax.axline([0,0],[1,1],linestyle='--',color='black',linewidth=1)
    
    fig.suptitle('Theoretical NWP volume vs Computed NWP volume')
    ax.set_ylabel('Pixel-tailed Vnw (ml)')
    ax.set_xlabel('Theoretical Vnw (ml)')
    ax.set_xlim(0,np.max(Vnw))
    ax.set_ylim(0,np.max(Vnw_computed))
    ax.legend()

    ## Discrepancy Vnw
    fig, ax = plt.subplots(figsize=(3.54,3.54),dpi=600)

    ax.scatter(np.arange(1,len(Vnw_computed)+1,1),Vnw_computed-Vnw,marker='o',alpha=0.5,linewidths=0.5,color='green')
    
    fig.suptitle('Discrepancy in NWP Volume')
    ax.set_ylabel('Vnw (ml)')

    return plt.show()

### Saturation field
def plot_map(type,snw,Iw,eff_snw,time,trap_eff=0):
        
    kwargs = {'format': '%.2f'}
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    
    fig,ax = plt.subplots(figsize=(3.54,3.54),dpi=600)

    ax.imshow(Iw,cmap='gray')
    Sof_field=np.ma.masked_where(snw<0.03,snw)
    max_val=np.round(np.max(Sof_field),2)
    Sof_field_img=ax.imshow(Sof_field,cmap='viridis',interpolation='nearest',vmin=0,vmax=max_val)

    plt.colorbar(Sof_field_img,ax=ax,orientation='horizontal',shrink=0.5,ticks=[0,float(max_val)],**kwargs).set_label(label='Pixel-wise Snw',size=5,weight='bold')
    
    if type == 'breakthrough':
        fig.suptitle('Domain Breakthrough',va='center',size='x-small',y=0.92)
        plt.text(x=500, y=2500,s='Saturation: '+ str(np.round(eff_snw,3)),fontsize=5,bbox=props)
        plt.text(x=500, y=2650,s= 'Breakthrough time: ' + time,fontsize=5,bbox=props)

        plt.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,
        left=False,        # ticks along the top edge are off
        labelbottom=False,
        labelleft=False) # labels along the bottom edge are off
    elif type == 'end':
        fig.suptitle('End of drainage',va='center',size= 'x-small',y=0.92)
        plt.text(x=500, y=2500,s='End of drainage saturation: '+ str(np.round(eff_snw,3)),fontsize=5,bbox=props)
        plt.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,
        left=False,        # ticks along the top edge are off
        labelbottom=False,
        labelleft=False) # labels along the bottom edge are off
    else:
        fig.suptitle('End of redistribution',va='center',size = 'x-small',y=0.92)
        plt.text(x=500, y=2500,s='End of redistribution saturation: '+ str(np.round(eff_snw,3)),fontsize=5,bbox=props)
        plt.text(x=500, y=2650,s='Trapping efficiency: ' + str(np.round(trap_eff,3)),fontsize=5,bbox=props)
        plt.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,
        left=False,        # ticks along the top edge are off
        labelbottom=False,
        labelleft=False) # labels along the bottom edge are off
        
    return plt.show(fig)

def height_eff_snw_plot(d,last,time,middle=15):
    y0, x0 = zip(*d[0])
    y1, x1 = zip(*d[middle])
    y2, x2 = zip(*d[last-1])

    fig,ax = plt.subplots(figsize=(3.54,3.54),dpi=600)
    
    ax.scatter(x0,y0,marker='x',s=5,label='t= 0')
    ax.plot(x0,y0,linestyle='--',linewidth=0.5)

    ax.scatter(x1,y1,marker='o',s=5,label='t1')
    ax.plot(x1,y1,linestyle='--',linewidth=0.5)

    ax.scatter(x2,y2,marker='x',s=5,label='t= '+ time)
    ax.plot(x2,y2,linestyle='--',linewidth=0.5)

    plt.ylabel('Height (cm)')
    plt.xlabel('Pixel-wise Snw')
    handles, labels = ax.get_legend_handles_labels()
    labels = ['t=0','t1','t= '+ time]
    ax.legend(handles,labels,loc='upper center', bbox_to_anchor=(0.5, -0.17),
          fancybox=True, shadow=True,ncol=6,fontsize='x-small')
    
    return plt.show()

def spatial_moments(time_break, time_dra, time_red, m00_break, m00_dra, m00_red,x_brk, z_brk, x_dra, z_dra, x_red, z_red):

    fig, ax = plt.subplots(figsize=(3.54,3.54),dpi=600)

    brk = ax.scatter(time_break,m00_break,color='skyblue',linewidths=0.1,alpha=0.6,s=5)
    dra = ax.scatter(time_dra,m00_dra,color='cornflowerblue',linewidths=0.1,alpha=0.7,s=5)
    red = ax.scatter(time_red,m00_red,color='darkblue',linewidths=0.1,alpha=0.8,s=5)

    ax.set_xlabel('Time, min')
    ax.set_ylabel('Detected plume volume (ml)')
    fig.suptitle('Zeroth moment')
    plt.legend((brk,dra,red),
           ('Domain Breakthrough', 'Drainage','Redistribution'),
           scatterpoints=1,
           loc='lower right',
           ncol=1,
           fontsize=8)
    
    fig, ax = plt.subplots(figsize=(3.54,3.54),dpi=600)

    brk = ax.scatter(time_break,x_brk,color='skyblue',linewidths=0.1,alpha=0.6,s=5)
    dra = ax.scatter(time_dra,x_dra,color='cornflowerblue',linewidths=0.1,alpha=0.7,s=5)
    red = ax.scatter(time_red,x_red,color='darkblue',linewidths=0.1,alpha=0.8,s=5)

    ax.set_xlabel('Time, min')
    ax.set_ylabel('X-coordinate (cm)')
    ax.set_ylim(1,60)
    fig.suptitle('First moment')
    plt.legend((brk,dra,red),
           ('Domain Breakthrough', 'Drainage','Redistribution'),
           scatterpoints=1,
           loc='lower right',
           ncol=1,
           fontsize=8)
    
    fig, ax = plt.subplots(figsize=(3.54,3.54),dpi=600)

    brk = ax.scatter(time_break,z_brk,color='skyblue',linewidths=0.1,alpha=0.6,s=5)
    dra = ax.scatter(time_dra,z_dra,color='cornflowerblue',linewidths=0.1,alpha=0.7,s=5)
    red = ax.scatter(time_red,z_red,color='darkblue',linewidths=0.1,alpha=0.8,s=5)

    ax.set_xlabel('Time, min')
    ax.set_ylabel('Z-coordinate (cm)')
    ax.set_ylim(1,40)
    fig.suptitle('First moment')
    plt.legend((brk,dra,red),
           ('Domain Breakthrough', 'Drainage','Redistribution'),
           scatterpoints=1,
           loc='lower right',
           ncol=1,
           fontsize=8)
    
    fig, ax = plt.subplots(figsize=(3.54,3.54),dpi=600)
    brk = ax.scatter(x_brk,z_brk,color='skyblue',linewidths=0.1,alpha=0.6,s=m00_break)
    dra = ax.scatter(x_dra,z_dra,color='cornflowerblue',linewidths=0.1,alpha=0.7,s=m00_dra)
    red = ax.scatter(x_red,z_red,color='darkblue',linewidths=0.1,alpha=0.8,s=m00_red)

    ax.set_xlabel('X-coordinate (cm)')
    ax.set_ylabel('Z-coordinate (cm)')
    ax.set_ylim(1,60)
    ax.set_xlim(1,60)
    fig.suptitle('Plume centroid saturation migration')
    plt.legend((brk,dra,red),
        ('Domain Breakthrough', 'Drainage','Redistribution'),
        scatterpoints=1,
        loc='upper right',
        ncol=1,
            fontsize=8)
    return plt.show()

def initial_residual_curves(snw_field_dra,snw_field_red,title):
    x = snw_field_dra[:,:,-1].flatten()
    y = snw_field_red[:,:,-1].flatten()

    fig, ax = plt.subplots(figsize=(3.54,3.54),dpi=600)
    ax.scatter(x,y,color='skyblue',linewidths=0.1,alpha=0.3,s=1)
    ax.axline([0,0],[1,1],linestyle='--',color='black',linewidth=1)
    fig.suptitle(title)
    ax.set_xlabel('Post Drainage Saturation')
    ax.set_ylabel('Post Redistribution Saturation')
    ax.set_xlim((0,1))
    ax.set_ylim((0,1))
    return plt.show()

def trapping_efficiency(x,y,tick_labels):
    fig,ax = plt.subplots(figsize= (3.54,3.54),dpi=600)
    ax.scatter(x,y,color='skyblue',linewidths=1,alpha=1,s=20)
    ax.plot(x,y,color='black',linewidth=0.4,linestyle='--')

    fig.suptitle('Trapping efficiency')
    ax.set_xlabel('Heterogeneity contrast')
    ax.set_ylabel('Sr/Si')
    ax.grid(axis='y',which='both',linestyle='--',linewidth=0.6)
    ax.set_xticklabels(tick_labels)

    return plt.show()

def video_generator (imgs,tiff_file_drainage,filename):
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
    
    f = r"Data_processed/"+filename
    writergif = animation.PillowWriter(fps=5) 
    ani.save(f, writer=writergif)

    return 

