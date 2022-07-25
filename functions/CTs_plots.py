#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import xarray as xr
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import cftime
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.colors as colors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import sys
directory = 'functions/'
sys.path.insert(0, directory)
import JK_functions
def plot_CT(CT, date):
    '''
    This function plots the 11 circulation types to a map
    ¡! Works only with Basemap module

    :param CT: xarray file of the eleven reduced circulation types

    '''
    CT = CT.sel(time = date)
    print('Would you like to provide the area to be plotted? \n yes/no')
    answer = input()
    if answer == 'yes':
        print('Input the Northernmost latitude:')
        lat_north = float(input())
        lat_north = CT.lat.sel(lat = lat_north, method = 'nearest')
        print('Input the Southernmost latitude:')
        lat_south = float(input())
        lat_south = CT.lat.sel(lat = lat_south, method = 'nearest')
        print('Input the Westernmost latitude (0 to -180):')
        lon_west = float(input())
        lon_west = CT.lon.sel(lon = lon_west, method = 'nearest')
        print('Input Easternmost longitude:')
        lon_east = float(input())
        lon_east = CT.lon.sel(lon = lon_east, method = 'nearest')
        CT = CT.sel(lat = slice(lat_north, lat_south), lon = slice(lon_west, lon_east))
        dif_x = lon_east - (lon_west)
        dif_y = lat_north - (lat_south)
        if dif_x > dif_y:
            size_x = 12
            size_y = size_x * (dif_y / dif_x)
        elif dif_x < dif_y:
            size_y = 12
            size_x = size_y * (dif_y / dif_x)
    else:
        print('Using default area')
        lat_north = float(CT.lat[0].values)
        lat_south = float(CT.lat[-1].values)
        lon_west  = float(CT.lon[0].values)
        lon_east  = float(CT.lon[-1].values)
        print(lat_south, 'to', lat_north, 'ºN' )
        print(lon_west, 'to', lon_east, 'ºE')
        size_x = 12
        size_y = 6

    CT = xr.where((CT.lat < 10) & (CT.lat > -10), np.nan, CT)
    #Defining colours to plot CTs
    colores = ListedColormap(["#7C7C77", "#17344F", "#0255F4","#0F78ED", "#9E09EE", "#F6664C",
                         "#F24E64", "#D3C42D", "#2FC698",
                         "#20E1D7", "#BD0000"])  #C, N, NW, W, SW, S, SE, E, NE, A, LF
    bounds = np.arange(-1,11,1)
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=12)
    lons, lats = np.meshgrid(CT.lon, CT.lat)
    #Defining size and map boundaries
    plt.gcf().clear()
    fig= plt.figure(figsize = (size_x,size_y))
    m = Basemap(projection='cyl',llcrnrlat=lat_south,urcrnrlat=lat_north,\
                    llcrnrlon=lon_west,urcrnrlon=lon_east,resolution='l')
    m.drawcoastlines()
    x,y = m(lons, lats)

    m.pcolor(x,y,CT,norm = norm,cmap = colores, alpha = 0.70)
    # m.drawparallels(np.arange(-90.,120.,30.))
    # m.drawmeridians(np.arange(0.,400.,40.))
    parallels = np.arange(-90.,120.,30.)
    # labels = [left,right,top,bottom]
    m.drawparallels(parallels,labels=[True,False,False,True])
    meridians = np.arange(0.,390.,30.)
    m.drawmeridians(meridians,labels=[True,False,False,True])
    #Legends
    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
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

    ax.legend(handles=legend_elements,loc = 'center', bbox_to_anchor=(1.1, 0.50),frameon = False, prop={'size': 14}) #X,Y, LENGTH, WIDHT

    plt.title(str(CT.time.values)[0:-19], size = 16)
    plt.tight_layout()
    plt.ioff()
    return fig

def plot_CT_MSLP(CT,date,source, filename_MSLP, variable='msl'):
    '''
    This function plots the 11 circulation types to a map
    ¡! Works only with Basemap module

    :param CT: xarray file of the eleven reduced circulation types
    :param date: a string in format "YYYY-MM-DD" indicating the date to be plotted
    :param source: a string indicating the source of the dataset. Either "REAN" or "GCM"
    :param filename_MSLP: a string indicating the location and name of the Mean Sea Level Pressure dataset

    '''
#     print('Provide the date to be plotted as "YYYY-MM-DD":')
#     date = str(input())
    MSLP = xr.open_dataset(filename_MSLP)
    try:
        MSLP = MSLP[variable]
    except KeyError:
        err_msg = (
            'Sea level pressure variable {:} cannot be found.'
            'Please provide the variable name of the xr.Dataset containing'
            ' sea level pressure.'
        )
        raise KeyError(err_msg.format(variable))

    # Convert Pa to hPa
    MSLP /= 100
    MSLP = MSLP.sel(time = date)
    CT = CT.sel(time = date)
    #Checking longitude coordinates to be - 180 to 180
    if source == 'REAN':
        lon_name = 'longitude'
    elif source == 'GCM':
        lon_name = 'lon'
    else:
        raise TypeError("Incorrect source, only 'REAN' or 'GCM' allowed")
    print('Checking if longitude coordinates are -180 to 180')
    MSLP = JK_functions.checking_lon_coords(MSLP, lon_name)


    print('Would you like to provide the area to be plotted? \n yes/no')
    answer = input()
    if answer == 'yes':
        print('Input the Northernmost latitude:')
        lat_north = float(input())
        lat_north = CT.lat.sel(lat = lat_north, method = 'nearest')
        print('Input the Southernmost latitude:')
        lat_south = float(input())
        lat_south = CT.lat.sel(lat = lat_south, method = 'nearest')
        print('Input the Westernmost latitude (0 to -180):')
        lon_west = float(input())
        lon_west = CT.lon.sel(lon = lon_west, method = 'nearest')
        print('Input Easternmost longitude:')
        lon_east = float(input())
        lon_east = CT.lon.sel(lon = lon_east, method = 'nearest')
        CT = CT.sel(lat = slice(lat_north, lat_south), lon = slice(lon_west, lon_east))
        dif_x = lon_east - (lon_west)
        dif_y = lat_north - (lat_south)
        if dif_x > dif_y:
            size_x = 12
            size_y = size_x * (dif_y / dif_x)
        elif dif_x < dif_y:
            size_y = 12
            size_x = size_y * (dif_y / dif_x)
    else:
        print('Using default area')
        lat_north = float(CT.lat[0].values)
        lat_south = float(CT.lat[-1].values)
        lon_west  = float(CT.lon[0].values)
        lon_east  = float(CT.lon[-1].values)
        print(lat_south, 'to', lat_north, 'ºN' )
        print(lon_west, 'to', lon_east, 'ºE')
        size_x = 12
        size_y = 6

    if source == 'REAN':
        MSLP = MSLP.sel(latitude = slice(lat_north, lat_south), longitude = slice(lon_west, lon_east))
        lat = MSLP.latitude
        lon = MSLP.longitude
        lon_list = list(lon)
        lat_list = list(lat)
    elif source == 'GCM':
        MSLP = MSLP.sel(lat = slice(lat_south, lat_north), lon = slice(lon_west, lon_east))
        lat = MSLP.lat
        lon = MSLP.lon
        lon_list = list(lon)
        lat_list = list(lat)
    else:
        raise TypeError("Source dataset not recognised!")

    CT = xr.where((CT.lat < 10) & (CT.lat > -10), np.nan, CT)

    order_lats = lat_list[0] - lat_list[-1]
    if order_lats < 0:
        MSLP = MSLP.reindex(lat=list(reversed(MSLP.lat)))
    else:
        None

    #Defining colours to plot CTs
    colores = ListedColormap(["#7C7C77", "#17344F", "#0255F4","#0F78ED", "#9E09EE", "#F6664C",
                         "#F24E64", "#D3C42D", "#2FC698", "#20E1D7", "#BD0000"])  #C, N, NW, W, SW, S, SE, E, NE, A, LF
    bounds = np.arange(-1,11,1)
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=12)
    lons, lats = np.meshgrid(CT.lon, CT.lat)
    #Defining size and map boundaries
    plt.gcf().clear()
    fig= plt.figure(figsize = (size_x,size_y))
    m = Basemap(projection='cyl',llcrnrlat=lat_south,urcrnrlat=lat_north,\
                    llcrnrlon=lon_west,urcrnrlon=lon_east,resolution='l')
    m.drawcoastlines()
    x,y = m(lons, lats)
    im1 = m.pcolor(x,y,CT,norm = norm,cmap = colores, alpha = 0.75)
    from scipy.ndimage.filters import gaussian_filter
    data3 = gaussian_filter(MSLP, sigma=.8)
    im2 = m.contour(x,y,data3, np.arange(1012-40, 1012+44, 4), colors = '0.10', linewidths = 0.5)
    plt.clabel(im2, im2.levels, inline=True,fmt ='%.0f', fontsize=10)
    parallels = np.arange(-90.,120.,30.)
    # labels = [left,right,top,bottom]
    m.drawparallels(parallels,labels=[True,False,False,True])
    meridians = np.arange(0.,390.,30.)
    m.drawmeridians(meridians,labels=[True,False,False,True])
    #Legends
    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
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


    ax.legend(handles=legend_elements,loc = 'center', bbox_to_anchor=(1.1, 0.50),frameon = False, prop={'size': 14}) #X,Y, LENGTH, WIDHT

    plt.title(str(CT.time.values)[0:-19], size = 16)
    plt.tight_layout()
    plt.ioff()
    return fig


def plot_CT_MSLP_globe(CT,date, source, filename_MSLP, variable='msl'):
    '''
    This function plots the 11 circulation types to a map
    ¡! Works only with Basemap module

    :param CT: xarray file of the eleven reduced circulation types
    :param date: a string in format "YYYY-MM-DD" indicating the date to be plotted
    :param source: a string indicating the source of the dataset. Either "REAN" or "GCM"
    :param filename_MSLP: a string indicating the location and name of the Mean Sea Level Pressure dataset

    '''
    MSLP = xr.open_dataset(filename_MSLP)
    try:
        MSLP = MSLP[variable]
    except KeyError:
        err_msg = (
            'Sea level pressure variable {:} cannot be found.'
            'Please provide the variable name of the xr.Dataset containing'
            ' sea level pressure.'
        )
        raise KeyError(err_msg.format(variable))

    # Convert Pa to hPa
    MSLP /= 100
    MSLP = MSLP.sel(time = date)
    CT = CT.sel(time = date)
    #Checking longitude coordinates to be - 180 to 180
    if source == 'REAN':
        lon_name = 'longitude'
    elif source == 'GCM':
        lon_name = 'lon'
    else:
        raise TypeError("Incorrect source, only 'REAN' or 'GCM' allowed")
    print('Checking if longitude coordinates are -180 to 180')
    MSLP = JK_functions.checking_lon_coords(MSLP, lon_name)
    print('Would you like to provide the central area where to plot? \n yes/no')
    answer = input()
    if answer == 'yes':
        print('Input the central latitude:')
        lat_central = float(input())
        print('Input the central longitude :')
        lon_central = float(input())
        lat_north = float(CT.lat[0].values)
        lat_south = float(CT.lat[-1].values)
        lon_west  = float(CT.lon[0].values)
        lon_east  = float(CT.lon[-1].values)
    else:
        print('Using default central area')
        lon_central = 0
        lat_central = 30
        print(lat_central, 'ºN Central latitude' )
        print(lon_central, 'ºE Central longitude')
        lat_north = float(CT.lat[0].values)
        lat_south = float(CT.lat[-1].values)
        lon_west  = float(CT.lon[0].values)
        lon_east  = float(CT.lon[-1].values)

    if source == 'REAN':
        MSLP = MSLP.sel(latitude = slice(lat_north, lat_south), longitude = slice(lon_west, lon_east))
        lat = MSLP.latitude
        lon = MSLP.longitude
        lon_list = list(lon)
        lat_list = list(lat)
    elif source == 'GCM':
        MSLP = MSLP.sel(lat = slice(lat_south, lat_north), lon = slice(lon_west, lon_east))
        lat = MSLP.lat
        lon = MSLP.lon
        lon_list = list(lon)
        lat_list = list(lat)
    else:
        raise TypeError("Source dataset not recognised!")
    CT = xr.where((CT.lat < 10) & (CT.lat > -10), np.nan, CT)

    order_lats = lat_list[0] - lat_list[-1]
    if order_lats < 0:
        MSLP = MSLP.reindex(lat=list(reversed(MSLP.lat)))
    else:
        MSLP = MSLP

    #Defining colours to plot CTs
    colores = ListedColormap(["#7C7C77", "#17344F", "#0255F4","#0F78ED", "#9E09EE", "#F6664C",
                         "#F24E64", "#D3C42D", "#2FC698",
                         "#20E1D7", "#BD0000"])  #C, N, NW, W, SW, S, SE, E, NE, A, LF
    bounds = np.arange(-1,11,1)
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=12)
    lons, lats = np.meshgrid(CT.lon, CT.lat)
    #Defining size and map boundaries
    # set perspective angle
    lat_viewing_angle = 50
    lon_viewing_angle = -10

    fig = plt.figure(figsize = (12,10))
    # call the basemap and use orthographic projection at viewing angle
    m = Basemap(projection='ortho',lon_0=lon_central,lat_0=lat_central,resolution='l')
    m.drawcoastlines()
    # m.etopo(scale=0.9, alpha=0.90)
    m.shadedrelief()
    # m.bluemarble(scale=0.5);
    x,y = m(lons, lats)
    im1 = m.pcolor(x,y,CT,norm = norm,cmap = colores, alpha = 0.75)
    from scipy.ndimage.filters import gaussian_filter
    data3 = gaussian_filter(MSLP, sigma=.8)
    im2 = m.contour(x,y,data3, np.arange(1012-40, 1012+44, 4), colors = '0.10', linewidths = 0.5)
    plt.clabel(im2, im2.levels, inline=True,fmt ='%.0f', fontsize=10)

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

    plt.legend(handles=legend_elements, loc='right', bbox_to_anchor=(0.65, 0.25, 0.5, 0.5),frameon = False, prop={'size': 14}) #X,Y, LENGTH, WIDHT
    plt.title(str(CT.time.values)[0:-19], size = 16, y = 1.05)

    plt.tight_layout()
    plt.ioff()
    return fig
