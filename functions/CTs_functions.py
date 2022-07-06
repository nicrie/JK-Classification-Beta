#!/usr/bin/env python
# coding: utf-8

# In[1]:
"""
@Author: Pedro Herrera-Lormendez
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.basemap import Basemap
import cftime
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.colors as colors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
def plot_CT(CT):
    #Defining colours to plot CTs
    colores = ListedColormap(["#7C7C77", "#000000", "#2179E4", "#1A49D7", "#8591FB",
                         "#FAF084", "#55DD9D", "#056704", "#548A80", 
                         "#AE64A6", "#BD0000"])
    bounds = np.arange(-1,11,1)
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=12)
    lons, lats = np.meshgrid(CT.lon, CT.lat)
    plt.figure(figsize = (10,8))
    m = Basemap(projection='cyl',llcrnrlat=30.5,urcrnrlat=70.5,\
                    llcrnrlon=-15.5,urcrnrlon=35.5,resolution='l')
    m.drawcoastlines()
    x,y = m(lons, lats)
    m.pcolormesh(x,y,CT,norm = norm,cmap = colores)
    cbar=m.colorbar(ticks=np.arange(-0.5,10.5,1.0))
    # cbar = fig.colorbar(cax, ticks=[-1, 0, 1])
    cbar.ax.set_yticklabels(['LF', 'A', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW',
                            'N', 'C']) 
    plt.title(str(CT.time.values)[0:-19])
    plt.tight_layout()
    plt.show()
    
def plot_CT_MSLP(CT,date,file_dir,source):
    '''
    This function plots the 11 circulation types to a map
    ¡! Works only with Basemap module
    
    :param CT: xarray file of the eleven reduced circulation types

    '''
#     print('Provide the date to be plotted as "YYYY-MM-DD":')
#     date = str(input())
    MSLP = xr.open_dataset(file_dir)
    MSLP = MSLP[list(MSLP.variables)[-1]]/100
    MSLP = MSLP.sel(time = date)[0]
    print('Would you like to provide the area to be plotted? \n yes/no')
    answer = input()
    if answer == 'yes':
        print('Input the Northernmost latitude:')
        lat_north = float(input())
        print('Input the Southernmost latitude:')
        lat_south = float(input())
        print('Input the Westernmost latitude (0 to -180):')
        lon_west = float(input())
        print('Input Easternmost longitude:')
        lon_east = float(input())
    else:
        print('Using default area')
        lat_north = float(CT.lat[0].values)
        lat_south = float(CT.lat[-1].values)
        lon_west  = float(CT.lon[0].values)
        lon_east  = float(CT.lon[-1].values)
        print(lat_south, 'to', lat_north, 'ºN' )
        print(lon_west, 'to', lon_east, 'ºE')
    
    if source == 'REAN':
        MSLP = MSLP.sel(latitude = slice(lat_north, lat_south), longitude = slice(lon_west, lon_east))
    elif source == 'GCM':
        MSLP = MSLP.sel(lat = slice(lat_south, lat_north), lon = slice(lon_west, lon_east))
    else:
        raise TypeError("Source dataset not recognised!")
    CT = CT.sel(time = date)[0]    
    CT = xr.where((CT.lat < 10) & (CT.lat > -10), np.nan, CT)
        
    #Defining colours to plot CTs
    colores = ListedColormap(["#7C7C77", "#17344F", "#0255F4","#0F78ED", "#9E09EE", "#F6664C",
                         "#F24E64", "#D3C42D", "#2FC698", "#20E1D7", "#BD0000"])  #C, N, NW, W, SW, S, SE, E, NE, A, LF  
    bounds = np.arange(-1,11,1)
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=12)
    lons, lats = np.meshgrid(CT.lon, CT.lat)
    #Defining size and map boundaries
    fig = plt.figure(figsize = (12,10))
    m = Basemap(projection='cyl',llcrnrlat=lat_south,urcrnrlat=lat_north,\
                    llcrnrlon=lon_west,urcrnrlon=lon_east,resolution='l')
    m.drawcoastlines()
    x,y = m(lons, lats)
    
    im1 = m.pcolor(x,y,CT,norm = norm,cmap = colores, alpha = 0.70)
    from scipy.ndimage.filters import gaussian_filter
    data3 = gaussian_filter(MSLP, sigma=.8)    
    im2 = m.contour(x,y,data3, np.arange(1012-40, 1012+44, 4), colors = '0.10', linewidths = 0.5)
    plt.clabel(im2, im2.levels, inline=True,fmt ='%.0f', fontsize=10)

    plt.title(str(CT.time.values)[0:-19], size = 16)
    #Legends


    legend_elements = [  Line2D([0], [0],marker ='o', color = '#7C7C77',markerfacecolor = None,linestyle='None', markersize = 17,label='LF'),
                         Line2D([0], [0], marker='o', color='#17344F', markerfacecolor=None, linestyle='None', markersize=17, label='A'),
                         Line2D([0], [0], marker='o', color='#0255F4', markerfacecolor=None, linestyle='None', markersize=17, label='NE'),
                         Line2D([0], [0], marker='o', color='#0F78ED', markerfacecolor=None, linestyle='None', markersize=17, label='E'),
                         Line2D([0], [0], marker='o', color='#9E09EE', markerfacecolor=None, linestyle='None', markersize=17, label='SE'),
                         Line2D([0], [0], marker='o', color='#F6664C', markerfacecolor=None, linestyle='None', markersize=17, label='S'),
                         Line2D([0], [0], marker='o', color='#F24E64', markerfacecolor=None, linestyle='None', markersize=17, label='SW'),
                         Line2D([0], [0], marker='o', color='#D3C42D', markerfacecolor=None, linestyle='None', markersize=17, label='W'),
                         Line2D([0], [0], marker='o', color='#2FC698', markerfacecolor=None, linestyle='None', markersize=17, label='NW'),
                         Line2D([0], [0], marker='o', color='#20E1D7', markerfacecolor=None, linestyle='None', markersize=17, label='N'),
                         Line2D([0], [0], marker='o', color='#BD0000', markerfacecolor=None, linestyle='None', markersize=17, label='C'),]


    fig.legend(handles=legend_elements, loc='right', bbox_to_anchor=(0.58, 0.07, 0.5, 0.5),frameon = False, prop={'size': 12}) #X,Y, LENGTH, WIDHT

    plt.tight_layout()
    plt.ioff()
    return fig


def eleven_CTs(CT):
    CT = xr.where( (CT == 11) | (CT==21) | (CT ==1), 1, CT) #NE
    CT = xr.where( (CT == 12) | (CT==22) | (CT==2), 2, CT) #E
    CT = xr.where( (CT == 13) | (CT==23) | (CT==3), 3, CT) #SE
    CT = xr.where( (CT == 14) | (CT==24) | (CT==4), 4, CT) #S
    CT = xr.where( (CT == 15) | (CT==25) | (CT==5), 5, CT) #SW
    CT = xr.where( (CT == 16) | (CT==26) | (CT==6), 6, CT) #W
    CT = xr.where( (CT == 17) | (CT==27) | (CT==7), 7, CT) #NW
    CT = xr.where( (CT == 18) | (CT==28) | (CT==8), 8, CT) #N
    # CT = xr.where(CT == -1, 0, CT)
    # CT = xr.where(CT ==  0, 10, CT)
    CT = xr.where(CT == 20, 9, CT) #C
    return(CT)

def seasonal_frelative_frequencies(filename, year_init, year_end):
    '''
    This function computes the seasonal relative frequencies of the JK derived 11 CTs
    
    '''
    #Reading the daily CTs dataset
    DS = xr.open_dataset(filename)
    CT = DS.CT.sel(time = slice(str(year_init)+'-03-01', str(year_end)+'-11-30'))
    del(DS)
    LF = xr.where(CT == -1, 1, np.nan)
    A = xr.where(CT ==  0, 1, np.nan)
    C = xr.where(CT == 20, 1, np.nan)
    NE = xr.where( (CT == 11) | (CT==21) | (CT ==1), 1, np.nan)
    E = xr.where( (CT == 12) | (CT==22) | (CT==2), 1, np.nan)
    SE = xr.where( (CT == 13) | (CT==23) | (CT==3), 1, np.nan)
    S = xr.where( (CT == 14) | (CT==24) | (CT==4), 1, np.nan)
    SW = xr.where( (CT == 15) | (CT==25) | (CT==5), 1, np.nan)
    W = xr.where( (CT == 16) | (CT==26) | (CT==6), 1, np.nan)
    NW = xr.where( (CT == 17) | (CT==27) | (CT==7), 1, np.nan)
    N = xr.where( (CT == 18) | (CT==28) | (CT==8), 1, np.nan)
    month_length = CT.time.dt.days_in_month
    len_seasons=month_length.resample(time = '1M').mean().resample(time = 'Q-FEB').sum()
    # weights = month_length.groupby('time.season') / month_length.groupby('time.season').sum()
    LF_seasons = ((LF.resample(time = 'Q-FEB').sum() / len_seasons) *100).groupby('time.season').mean(dim = 'time')

    A_seasons   = ((A.resample(time = 'Q-FEB').sum() / len_seasons) *100).groupby('time.season').mean(dim = 'time')

    C_seasons   = ((C.resample(time = 'Q-FEB').sum() / len_seasons) *100).groupby('time.season').mean(dim = 'time')

    N_seasons  = ((N.resample(time = 'Q-FEB').sum() / len_seasons) *100).groupby('time.season').mean(dim = 'time')
    NE_seasons = ((NE.resample(time = 'Q-FEB').sum() / len_seasons) *100).groupby('time.season').mean(dim = 'time')
    E_seasons  = ((E.resample(time = 'Q-FEB').sum() / len_seasons) *100).groupby('time.season').mean(dim = 'time')
    SE_seasons = ((SE.resample(time = 'Q-FEB').sum() / len_seasons) *100).groupby('time.season').mean(dim = 'time')
    S_seasons  = ((S.resample(time = 'Q-FEB').sum() / len_seasons) *100).groupby('time.season').mean(dim = 'time')
    SW_seasons = ((SW.resample(time = 'Q-FEB').sum() / len_seasons) *100).groupby('time.season').mean(dim = 'time')
    W_seasons  = ((W.resample(time = 'Q-FEB').sum() / len_seasons) *100).groupby('time.season').mean(dim = 'time')
    NW_seasons = ((NW.resample(time = 'Q-FEB').sum() / len_seasons) *100).groupby('time.season').mean(dim = 'time')
    
    CTs = {'LF':LF_seasons, 
       'A': A_seasons, 
      'C':C_seasons, 
      'NE':NE_seasons, 'E':E_seasons, 'SE':SE_seasons, 'S':S_seasons, 'SW':SW_seasons, 'W':W_seasons, 'NW':NW_seasons, 'N':N_seasons}
    seasons_labels=NW_seasons.season
    lat_vals = NW_seasons.lat
    lon_vals = NW_seasons.lon
    seasonal = []
    for i in CTs:
        seasonal.append(CTs[i])
    seasonal = np.array(seasonal)
    seasonal_xr=xr.DataArray(seasonal,
                  coords = {'CT': list(CTs.keys()),                        
                            'season': seasons_labels,
                           'lat': lat_vals,
                           'lon': lon_vals},
                  dims = ['CT', 'season', 'lat', 'lon'],
                  attrs=dict(
                        description="Seasonal relative frequencies period " + str(year_init)+'-'+str(year_end),
                        units="%",
                  ) )

    seasonal_xr.name = 'rel_freq'
    return(seasonal_xr)



