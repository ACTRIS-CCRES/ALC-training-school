

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from datetime import date, timedelta
import matplotlib.image as mpimg

from pvlib.location import Location




def from_df_to_pivot(df, var, resample):
    ''' function to reshape dataframes into easier form to analyse 
    '''
## dataframe tel que les indices sont : temps station range
    df = df[[var]]
    #print (df)
    if (var == "RBCS_tR") | (var == "RBCS") :
        df.reset_index(level=1, drop=True, inplace=True)
        df.reset_index(level=0, inplace=True)
        df.reset_index(level=0, inplace=True)
        df_pivot = df.pivot(index="range", columns="time", values=var)
    if var == "rcs_0":
        df.reset_index(level=0, inplace=True)
        df.reset_index(level=0, inplace=True)
        df_pivot = df.pivot(index="range", columns="time", values=var)
    if var == "beta":
        df.reset_index(level=0, inplace=True)
        df.reset_index(level=0, inplace=True)
        df.reset_index(level=0, drop=True, inplace=True)
        df_pivot = df.pivot(index="altitude", columns="time", values=var)
    if var == "sigma_w":
        df.reset_index(level=0, inplace=True)
        df.reset_index(level=0, inplace=True)
        df_pivot = df.pivot(index="altitude", columns="time", values=var)
    

    if resample == 1 :
        # 1. Si tes colonnes sont des timestamps en texte, convertis-les en datetime :
        # df_pivot.columns = pd.to_datetime(df_pivot.columns)

        # 2. On remet les pas de temps en lignes (melt ou transpose)
        df_long = df_pivot.transpose()

        # 3. On ajoute l’info temporelle comme index
        df_long.index.name = "time"

        # 4. On fait le resample sur 5 minutes
        df_pivot = df_long.resample("5min").mean()  
        df_pivot = df_pivot.T  
    
    return df_pivot





def plot_overlap_corrections (overlap_correction_palaiseau, overlap_correction_payerne, overlap_correction_granada, 
                              pivot_raw_signal_chm_palaiseau, pivot_raw_signal_chm_payerne, pivot_raw_signal_chm_granada, 
                              pivot_corrected_signal_chm_palaiseau, pivot_corrected_signal_chm_payerne, pivot_corrected_signal_chm_granada) :
    
    fig, axs = plt.subplots(1, 4, figsize=(10, 3), gridspec_kw={'width_ratios': [1.5, 1, 1, 1]})

    fig.suptitle("CHM15K corrections and signals", fontsize=20, y=1.15)

    positions = [
        [0.05, 0.1, 0.25, 0.8],  # First subplot (left)
        [0.38, 0.1, 0.15, 0.8],  # 2nd subplot (wider)
        [0.55, 0.1, 0.15, 0.8],  # 3rd subplot 
        [0.72, 0.1, 0.15, 0.8]   # 4th subplot
    ]

    # Appliquer les nouvelles positions
    for ax, pos in zip(axs, positions):
        ax.set_position(pos)


    # plot corrections
    axs[0].set_title("Correction models", fontsize=15)
    axs[0].set_xlabel("range [magl]", fontsize=12)

    lim = 700

    rel_diff = np.polyval([overlap_correction_palaiseau.a,overlap_correction_palaiseau.b], 20) # here, we use an internal temperature = 20°C
    ov_rec_all = 1./ (rel_diff/100/overlap_correction_palaiseau.overlap_ref + 1./overlap_correction_palaiseau.overlap_ref)
    axs[0].plot(ov_rec_all.loc[lim-600:lim], label ='Palaiseau', color='k')

    rel_diff = np.polyval([overlap_correction_payerne.a,overlap_correction_payerne.b], 20)
    ov_rec_all = 1./ (rel_diff/100/overlap_correction_payerne.overlap_ref + 1./overlap_correction_payerne.overlap_ref)
    axs[0].plot(ov_rec_all.loc[lim-600:lim], label ='Payerne')

    rel_diff = np.polyval([overlap_correction_granada.a,overlap_correction_granada.b], 20)
    ov_rec_all = 1./ (rel_diff/100/overlap_correction_granada.overlap_ref + 1./overlap_correction_granada.overlap_ref)
    axs[0].plot(ov_rec_all.loc[lim-600:lim], label ='Granada')

    axs[0].plot(overlap_correction_payerne.overlap_ref.loc[lim-600:lim],  "--", label ='Reference')

    axs[0].legend(loc = "upper left")


    # plot vertical profiles
    axs[1].plot(pivot_raw_signal_chm_palaiseau.iloc[:60, :].mean(axis=1), pivot_raw_signal_chm_palaiseau.iloc[:60, :].index, label = 'raw signal', color='gray')
    axs[1].plot(3.08e11*pivot_corrected_signal_chm_palaiseau.iloc[:30, :].mean(axis=1), pivot_corrected_signal_chm_palaiseau.iloc[:30, :].index, label = 'corrected signal', color='k')
    y1, y2 = 0, 225  # Définition des bornes
    axs[1].axhspan(y1, y2, color='gray', alpha=0.3)  # Alpha pour la transparence
    axs[1].set_xlim([-100000,400000])
    axs[1].set_ylabel("range [magl]", fontsize=12)
    axs[1].set_title("Palaiseau", fontsize=15)
    axs[1].legend(loc='upper right', fontsize=8)


    axs[2].plot(pivot_raw_signal_chm_payerne.iloc[:60, :].mean(axis=1), pivot_raw_signal_chm_payerne.iloc[:60, :].index, label = 'raw signal', color='skyblue')
    axs[2].plot(2.42e11*pivot_corrected_signal_chm_payerne.iloc[:30, :].mean(axis=1), pivot_corrected_signal_chm_payerne.iloc[:30, :].index, label = 'corrected signal',color='blue')
    y1, y2 = 0, 254  # Définition des bornes
    axs[2].axhspan(y1, y2, color='gray', alpha=0.3)
    axs[2].set_xlim([-100000,400000])
    axs[2].set_xlabel("normalized range corrected signal [m^2*counts/s]", fontsize=12)
    axs[2].set_title("Payerne", fontsize=15)
    axs[2].set_yticklabels([])  
    axs[2].legend(loc='upper right', fontsize=8)


    axs[3].plot(pivot_raw_signal_chm_granada.iloc[:60, :].mean(axis=1), pivot_raw_signal_chm_granada.iloc[:60, :].index, label = 'raw signal', color='gold')
    axs[3].plot(3.63e11*pivot_corrected_signal_chm_granada.iloc[:30, :].mean(axis=1), pivot_corrected_signal_chm_granada.iloc[:30, :].index, label = 'corrected signal', color='darkorange')
    y1, y2 = 0, 240  # Définition des bornes
    axs[3].axhspan(y1, y2, color='gray', alpha=0.3)  
    axs[3].set_xlim([-100000,400000])
    axs[3].set_title("Granada", fontsize=15)
    axs[3].set_yticklabels([])  
    axs[3].legend(loc='upper right', fontsize=8)





def plot_MLH(date_, field1, field2, site1, site2, df_MLH, vmin1, vmin2, vmax1, vmax2, MLH_, zoom, cmap1):

    ''' function to plot MLH on top of aerosols backscatter fields 
    date_ = pd.datetime()
    field1 = 1st field to plot, as a pivot dataframe 
    field2 = ...
    df_MLH = dataframe containing MLH of 1st field and 2nd field 
    '''
    
    fig = plt.figure(figsize=(12, 6))  

    # Define GridSpec with space for colorbars on the right
    gs = gridspec.GridSpec(2, 2, width_ratios=[20, 1], height_ratios=[1, 1])
    gs.update(hspace=0.4)  # Increase vertical spacing between plots


    # --- Plot limits ----
    if zoom == 1 :
        field1 = field1.loc[:1000].iloc[:, 0:120]
        field2 = field2.loc[:1000].iloc[:, 0:120]

    # --- First subplot: Raw signal ---
    ax0 = plt.subplot(gs[0, 0])
    ax0.set_title(f"{date_.strftime('%Y-%m-%d')} {site1}", fontsize=15)
    
    img0 = ax0.pcolor( 
        field1.columns,  
        field1.index,   
        field1,  
        vmin=vmin1,
        vmax=vmax1,
        cmap=cmap1,
    )
    ax0.set_ylabel("Range [magl]")

    # --- Second subplot: Corrected and calibrated signal ---
    ax1 = plt.subplot(gs[1, 0], sharex=ax0)  # Align x-axis
    ax1.set_title(f"{date_.strftime('%Y-%m-%d')} {site2}", fontsize=15)
    

    img1 = ax1.pcolor(
        field2.columns,  
        field2.index,    
        field2,  
        vmin=vmin2,
        vmax=vmax2,
        cmap="viridis",
    )
    ax1.set_ylabel("Range [magl]")
    
    # --- Single Colorbars spanning both subplots ---
    #cbar_ax0 = fig.add_axes([0.82, 0.15, 0.02, 0.7])  # (left, bottom, width, height)
    #cbar0 = plt.colorbar(img0, cax=cbar_ax0)
    #cbar0.set_label("Variance [m².s⁻²]")

    cbar_ax0 = fig.add_axes([0.82, 0.15, 0.02, 0.7])  # Second colorbar next to the first
    cbar0 = plt.colorbar(img0, cax=cbar_ax0)
    #cbar0.set_label("Normalized range corrected signal [m⁻¹.sr⁻¹]")

    cbar_ax1 = fig.add_axes([0.90, 0.15, 0.02, 0.7])  # Second colorbar next to the first
    cbar1 = plt.colorbar(img1, cax=cbar_ax1)
    cbar1.set_label("Normalized range corrected signal [m⁻¹.sr⁻¹]")

   
    # --- Plot MLH data ---
    if MLH_ == 1 :
        indeces_mlh = pd.date_range(
            start=f"{date_.strftime('%Y-%m-%d')} 00:00:00",
            end=f"{date_.strftime('%Y-%m-%d')} 00:23:45",
            periods=96
        )
        
        MLH_field1 = df_MLH[f"MLH_{site1}"][df_MLH.index.date == indeces_mlh[0].date()]
        MLH_field2 = df_MLH[f"MLH_{site2}"][df_MLH.index.date == indeces_mlh[0].date()]
        indeces_to_plot = MLH_field2.index

        max_mlh = MLH_field2.max() + 400

        if ((max_mlh < 1000) | (np.isnan(max_mlh))):
            max_mlh = 2500

        if zoom == 1 :
            max_mlh = 1000

        ax0.plot(indeces_to_plot, MLH_field1, '.', color='red', markersize=12, label = f"MLH {site1}")
        ax0.plot(indeces_to_plot, MLH_field2, '.', markerfacecolor='white', markeredgecolor='black', markersize=10, label = f"MLH {site2}")
        ax0.set_ylim([0,max_mlh])

        ax1.plot(indeces_to_plot, MLH_field1, '.', color='red', markersize=12)
        ax1.plot(indeces_to_plot, MLH_field2, '.', markersize=10, markerfacecolor='white', markeredgecolor='black')
        ax1.set_ylim([0,max_mlh])
        
        if zoom == 1 :
            ax1.set_xlim([indeces_to_plot[0], indeces_to_plot[40]])        
            ax1.set_xlim([indeces_to_plot[0], indeces_to_plot[40]])


        ax0.legend(loc="upper left")







def plot_MLH_week(date_1, date_2, field1, field2, site1, site2, df_MLH, vmin1, vmin2, vmax1, vmax2, MLH_, zoom, cmap1):

    ''' function to plot MLH on top of aerosols backscatter fields 
    date_ = pd.datetime()
    field1 = 1st field to plot, as a pivot dataframe 
    field2 = ...
    df_MLH = dataframe containing MLH of 1st field and 2nd field 
    '''
    
    fig = plt.figure(figsize=(12, 6))  

    # Define GridSpec with space for colorbars on the right
    gs = gridspec.GridSpec(2, 2, width_ratios=[20, 1], height_ratios=[1, 1])
    gs.update(hspace=0.4)  # Increase vertical spacing between plots


    # --- Plot limits ----
    if zoom == 1 :
        field1 = field1.loc[:1000].iloc[:, 0:120]
        field2 = field2.loc[:1000].iloc[:, 0:120]

    # --- First subplot: Raw signal ---
    ax0 = plt.subplot(gs[0, 0])
    #ax0.set_title(f"{date_.strftime('%Y-%m-%d')} {site1}", fontsize=15)
    
    img0 = ax0.pcolor( 
        field1.columns,  
        field1.index,   
        field1,  
        vmin=vmin1,
        vmax=vmax1,
        cmap=cmap1,
    )
    ax0.set_ylabel("Range [magl]")

    # --- Second subplot: Corrected and calibrated signal ---
    ax1 = plt.subplot(gs[1, 0], sharex=ax0)  # Align x-axis
    #ax1.set_title(f"{date_.strftime('%Y-%m-%d')} {site2}", fontsize=15)
    

    img1 = ax1.pcolor(
        field2.columns,  
        field2.index,    
        field2,  
        vmin=vmin2,
        vmax=vmax2,
        cmap="viridis",
    )
    ax1.set_ylabel("Range [magl]")
    
    # --- Single Colorbars spanning both subplots ---
    #cbar_ax0 = fig.add_axes([0.82, 0.15, 0.02, 0.7])  # (left, bottom, width, height)
    #cbar0 = plt.colorbar(img0, cax=cbar_ax0)
    #cbar0.set_label("Variance [m².s⁻²]")

    cbar_ax0 = fig.add_axes([0.82, 0.15, 0.02, 0.7])  # Second colorbar next to the first
    cbar0 = plt.colorbar(img0, cax=cbar_ax0)
    #cbar0.set_label("Normalized range corrected signal [m⁻¹.sr⁻¹]")

    cbar_ax1 = fig.add_axes([0.90, 0.15, 0.02, 0.7])  # Second colorbar next to the first
    cbar1 = plt.colorbar(img1, cax=cbar_ax1)
    cbar1.set_label("Normalized range corrected signal [m⁻¹.sr⁻¹]")

   
    # --- Plot MLH data ---
    if MLH_ == 1 :
        indeces_mlh = pd.date_range(
            start=f"{date_1.strftime('%Y-%m-%d')} 00:00:00",
            end=f"{date_2.strftime('%Y-%m-%d')} 00:23:45",
            periods=576
        )
        
        MLH_field1 = df_MLH[f"MLH_{site1}"][(df_MLH.index.date >= indeces_mlh[0].date()) & (df_MLH.index.date <= indeces_mlh[-1].date())]
        MLH_field2 = df_MLH[f"MLH_{site2}"][(df_MLH.index.date >= indeces_mlh[0].date()) & (df_MLH.index.date <= indeces_mlh[-1].date())]
        indeces_to_plot = MLH_field2.index

        max_mlh = MLH_field2.max() + 400

        if ((max_mlh < 1000) | (np.isnan(max_mlh))):
            max_mlh = 2500

        if zoom == 1 :
            max_mlh = 1000

        ax0.plot(indeces_to_plot, MLH_field1, '.', color='red', markersize=8, label = f"MLH {site1}")
        ax0.plot(indeces_to_plot, MLH_field2, '.', markerfacecolor='white', markeredgecolor='black', markersize=8, label = f"MLH {site2}")
        ax0.set_ylim([0,max_mlh])

        ax1.plot(indeces_to_plot, MLH_field1, '.', color='red', markersize=8)
        ax1.plot(indeces_to_plot, MLH_field2, '.', markersize=8, markerfacecolor='white', markeredgecolor='black')
        ax1.set_ylim([0,max_mlh])
        
        if zoom == 1 :
            ax1.set_xlim([indeces_to_plot[0], indeces_to_plot[40]])        
            ax1.set_xlim([indeces_to_plot[0], indeces_to_plot[40]])


        ax0.legend(loc="upper right")















    ### FUNCTIONS ###

## fonction get sunrise and sunset 

def get_sr_ss (site, dict_MLH, dict_lat, dict_lon) -> pd.DataFrame:
    """get sunrise and sunset and add them in dataframe 

    Args:
        site : containing L2A or L2B 

    Returns:
        dataframe: completed with sr and ss
    """
        
    lat, lon = dict_lat[f"lat_{site}"], dict_lon[f"lon_{site}"] 
    start_date = dict_MLH[f"MLH_{site}"].index.date[0]
    end_date = dict_MLH[f"MLH_{site}"].index.date[int(len(dict_MLH[f"MLH_{site}"])-2)]

    location = Location(lat, lon, tz='UTC')

    date_ = start_date - timedelta(days=1)
    sunrise = []
    sunset = []

    while date_ <= (end_date + timedelta(days=1)) :
        date_range = pd.date_range(start=date_, periods=1, freq='D')
        df_test = pd.DataFrame(index=date_range)
    
        # Calcul des heures de lever, coucher et transit du soleil
        sun_times = location.get_sun_rise_set_transit(df_test.index.tz_localize('UTC'))
        schedule = {'sunrise':pd.Timestamp(sun_times["sunrise"].values[0]), 'sunset':pd.Timestamp(sun_times["sunset"].values[0])}
        
        #schedule = sun.sun_utc(date_, lat, lon)
        sunrise1 = (schedule['sunrise'].second + 60*schedule['sunrise'].minute + 3600*schedule['sunrise'].hour)/3600. #en heure
        sunrise1 = 0.25 * (round(4.0 * sunrise1))  #en heure arrondie au quart d'heure le plus proche
        sunset1 = (schedule['sunset'].second + 60*schedule['sunset'].minute + 3600*schedule['sunset'].hour)/3600.
        sunset1 = 0.25 * (round(4.0 * sunset1))
        sunrise.append(sunrise1)
        sunset.append(sunset1)
        date_ = date_ + timedelta(days=1)
        

    ## TIME DAY REL SR SS

    data_day=96
    time = [i/4. for i in range(data_day)]*(end_date - start_date).days
    dt=[]

    for j in range(1, len(sunrise)-1):
        for i in range(data_day):
            if time[i] < sunset[j] :
                dt.append(time[i]-sunrise[j])
            else :
                dt.append(-(24-time[i] + sunrise[j+1]))
            
    dt = dt[:len(dict_MLH[f"MLH_{site}"])]
    dict_MLH[f"MLH_{site}"].insert(0,'time_relSR', dt)


    ss = []
    for j in range(len(sunset)-2):
        for i in range(data_day):
            if time[i] < sunrise[j+1] :
                ss.append((24-sunset[j] + time[i]))
            else:
                ss.append(time[i]-sunset[j+1])
            
    ss = ss[:len(dict_MLH[f"MLH_{site}"])]        
    dict_MLH[f"MLH_{site}"].insert(1,'time_relSS', ss)


    n=1
    day=[1]

    for i in range(1, len(dict_MLH[f"MLH_{site}"])):
        if dt[i-1]>0 and dt[i]<0 :
            day.append(n+1)
            n=n+1
        else :
            day.append(n)           #day relative to sunrise/sunset
        
    dict_MLH[f"MLH_{site}"].insert(2, 'day', day)

    return dict_MLH[f"MLH_{site}"]







def get_med (site, data, dict_MLH, dict_lat, dict_lon) -> pd.DataFrame :
    
    # by season

    lim = 0 
    seas = ((dict_MLH[f"MLH_{site}"].index.month >= 6) & (dict_MLH[f"MLH_{site}"].index.month <= 8))
    dict_MLH[f"MLH_{site}_jja"] = dict_MLH[f"MLH_{site}"][seas].groupby('time_relSR', as_index=False).agg(mlh = (f"{data}" , lambda x : x.median() if ((~np.isnan(x)).sum() > lim) else None), 
                                                                                                    q1 = (f"{data}" , lambda x : x.quantile(0.25) if ((~np.isnan(x)).sum() > lim) else None), 
                                                                                                    q3 = (f"{data}" , lambda x : x.quantile(0.75) if ((~np.isnan(x)).sum() > lim) else None), 
                                                                                                    max_mlh = (f"{data}" , lambda x : np.max(x) if ((~np.isnan(x)).sum() > lim) else None), 
                                                                                                    nsamples = (f'{data}', lambda x : (~np.isnan(x)).sum()))


    return dict_MLH[f"MLH_{site}_jja"]





def plot_aerosol_backscatter (df) :
    ''' 
    Show an aerosol backscatter signal 
    '''

    #df = pivot_raw_signal_chm_palaiseau.copy()

    fig, ax = plt.subplots(figsize=(12, 6))

    img0 = ax.pcolor(
        df.columns,   # X
        df.index,     # Y
        df.values,    # Z
        vmin=0,
        vmax=3 * 10**5,
        cmap="viridis",
    )

    # Titre et labels
    ax.set_title("Aerosol Backscatter signal, SIRTA 04 July 2022", fontsize=15)
    ax.set_xlabel("Time")
    ax.set_ylabel("Range [magl]")

    # Colorbar
    cbar = fig.colorbar(img0, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized range corrected signal [m⁻¹ sr⁻¹]")