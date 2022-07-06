#!/usr/bin/env python
# coding: utf-8

# In[ ]:
"""
@Author: Pedro Herrera-Lormendez
"""

import numpy as np
import pandas as pd
import xarray as xr
#Importing the directory where the neccesary functions are located
import JK_functions #Functions that help compute the CTs

def JK_classification(filename, source):
    
    '''
    
    @authors: Herrera-Lormendez, Pedro & John, Amal
    TU Bergakademie Freiberg and CNRS/Météo-France
    
    This computes the gridded Jenkinson-Collison circulation types
    derived from the original Lamb Weather Types Classification
    
    Computation of circulation types employs Mean Sea Level Pressure data
    
    :param filename: str. name and directory of the MSLP file
    :param source: str. Use "REAN" for ERA5 and ERA20C reanalysis and "GCM" when using GCMs
    :return: grided circulation types data as an xarray file
    '''
    if type(filename) == str:
        print('Reading filename: ', filename)
        #Reading the file
        DS = xr.open_dataset(filename)
        mslp = DS[list(DS.variables)[-1]]/100 #Reads the MSLP variable and converts to hPa
        if source == 'REAN': #ERA5 or ERA20C reanalyses
            DS.close()
            lon_name = 'longitude'
            lat_name = 'latitude'
            print('Do you wish to provide the time frame for the computation? (yes/no)')
            answer_time = input()
            if answer_time == 'yes':
                print('Time 0:',str(mslp.time[0].values))
                print('Time n-1:', str(mslp.time[-1].values))
                print('Provide starting time in YYYY-MM-DD format:')
                time_init = input()
                print('Provide ending time in YYYY-MM-DD format:')
                time_end = input()
            elif answer_time == 'no':
                time_init = str(mslp.time[0].values)
                time_end = str(mslp.time[-1].values)
                print('Using default time period from ' +  str(time_init) + ' to ' +  str(time_end))                
                
            else:
                raise TypeError("Incorrect answer! Only 'yes' and 'no' is allowed")
            
            #Cropping MSLP data in the time coordinate. 
            mslp =mslp.sel(time = slice(time_init,time_end))
            #Checking longitude coordinates to be - 180 to 180, if not (0 to 360) then fixed
            print('Checking if longitude coordinates are -180 to 180')
            mslp = JK_functions.checking_lon_coords(mslp, lon_name)

            #Computing factors based on grid size and spacing
            dif_lon = float((mslp.longitude[1]-mslp.longitude[0])) #Longitude grid spacing
            dif_lat = float((mslp.latitude[1]-mslp.latitude[0])) #Latitude grid spacing
            factor_lat = float(np.abs(1/dif_lat))
            factor_lon = float(np.abs(1/dif_lon))
            f_lat1 = int(10*factor_lat)
            f_lat2 = int(-10*factor_lat)
            f_lon1 = int(15*factor_lon)
            f_lon2 = int(-15*factor_lon)
            print('does your data covers the whole Globe? (yes/no)')
            answer_globe = input()  
            if answer_globe == 'yes':
                psl_area=mslp[:,f_lat1:f_lat2, :] 
            elif answer_globe == 'no':
                psl_area=mslp[:,f_lat1:f_lat2, f_lon1:f_lon2]
            else:
                raise TypeError("Incorrect answer! Only 'yes' and 'no' is allowed")
            #Extracting values of longitude, latitude and time coordinates
            lat = psl_area.latitude
            lon = psl_area.longitude
            time = psl_area.time.values
            time_len = len(time)
            lon_list = list(lon)
            lat_list = list(lat)
            len_lon = len(lon_list)
            len_lat = len(lat_list)
            
            ds = []
            #Computing latitude dependent constants
            print('Calculating latitude dependant constants ☀︎')
            lat_central = lat
            phi = lat_central            
            constants = JK_functions.constants(phi, lon)
            sc = constants[0]
            zwa = constants[1]
            zwb = constants[2]
            zsc = constants[3]
            print('Checking time formats ☽')
            #Checking the time coordinate values, since some models use different calendars
            if type(time[0]) == np.datetime64:
                time_pd = pd.to_datetime(time)
                dates = [pd.to_datetime(str(time_pd[t].year) + '-' + str(time_pd[t].month) + '-' + str(time_pd[t].day), 
                                            format = '%Y%m%d',errors = 'ignore') for t in range(len(time))]
            else:
                dates = [pd.to_datetime(str(time[t].year) + '-' + str(time[t].month) + '-' + str(time[t].day), 
                                    format = '%Y%m%d',errors = 'ignore') for t in range(len(time))]

            #Extracting the 16-gridded values of MSLP
            print('Extracting 16 gridpoints ●')
            if answer_globe == 'yes':
                gridpoints = JK_functions.extracting_gridpoints_rean_globe(mslp, lat, lon)
                p1  = gridpoints[0]
                p2  = gridpoints[1]
                p3  = gridpoints[2]
                p4  = gridpoints[3]
                p5  = gridpoints[4]
                p6  = gridpoints[5]
                p7  = gridpoints[6]
                p8  = gridpoints[7]
                p9  = gridpoints[8]
                p10 = gridpoints[9]
                p11 = gridpoints[10]
                p12 = gridpoints[11]
                p13 = gridpoints[12]
                p14 = gridpoints[13]
                p15 = gridpoints[14]
                p16 = gridpoints[15]
                del(gridpoints)
            elif answer_globe == 'no':
                gridpoints = JK_functions.extracting_gridpoints_rean_area(mslp, lat, lon)
                p1  = gridpoints[0]
                p2  = gridpoints[1]
                p3  = gridpoints[2]
                p4  = gridpoints[3]
                p5  = gridpoints[4]
                p6  = gridpoints[5]
                p7  = gridpoints[6]
                p8  = gridpoints[7]
                p9  = gridpoints[8]
                p10 = gridpoints[9]
                p11 = gridpoints[10]
                p12 = gridpoints[11]
                p13 = gridpoints[12]
                p14 = gridpoints[13]
                p15 = gridpoints[14]
                p16 = gridpoints[15]
                del(gridpoints)
            else:
                raise TypeError("Incorrect answer! Only 'yes' and 'no' is allowed")                

            print('Computing flow terms ☈')
            #Computing equations of flows and vorticity            
            flows = JK_functions.flows_rean(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, sc, zwa, zsc, zwb, lat, lon, time, mslp)
            W  = flows[0] #Westerly flow
            S  = flows[1] #Southerly flow
            F  = flows[2] #Resultant flow
            ZW = flows[3] #Westerly shear vorticity
            ZS = flows[4] #Southerly shear vorticity
            Z  = flows[5] #Total shear vorticity   

        else: #CMIP6 datasets
            institution_id = DS.institution_id
            experiment_id = DS.experiment_id
            source_id = DS.source_id
            DS.close()
            lon_name = 'lon'
            lat_name = 'lat'            
            print('Do you wish to provide the time frame for the computation? (yes/no)')
            answer_time = input()
            if answer_time == 'yes':
                print('Time 0:',str(mslp.time[0].values))
                print('Time n-1:', str(mslp.time[-1].values))                
                print('Provide starting time in YYYY-MM-DD format:')
                time_init = input()
                print('Provide ending time in YYYY-MM-DD format:')
                time_end = input()
            elif answer_time == 'no':
                time_init = str(mslp.time[0].values)
                time_end = str(mslp.time[-1].values)
                print('Using default time period from ' +  str(time_init) + ' to ' +  str(time_end))                
                
            else:
                raise TypeError("Incorrect answer! Only 'yes' and 'no' is allowed")
                
            mslp = mslp.sel(time = slice(time_init,time_end))
            #Checking longitude coordinates to be - 180 to 180
            print('Checking if longitude coordinates are -180 to 180')
            mslp = JK_functions.checking_lon_coords(mslp, lon_name)

            #Computing factors based on grid size
            dif_lon = float((mslp.lon[1]-mslp.lon[0]))
            dif_lat = float((mslp.lat[1]-mslp.lat[0]))

            factor_lat = float(np.abs(1/dif_lat))
            factor_lon = float(np.abs(1/dif_lon))
            f_lat1 = int(10*factor_lat)
            f_lat2 = int(-10*factor_lat)
            
            f_lon1 = int(15*factor_lon)
            f_lon2 = int(-15*factor_lon)
            print('does your data covers the whole Globe? (yes/no)')
            answer_globe = input()
            if answer_globe == 'yes':
                psl_area=mslp[:,f_lat1:f_lat2, :] 
            elif answer_globe == 'no':
                psl_area=mslp[:,f_lat1:f_lat2, f_lon1:f_lon2]
            else:
                raise TypeError("Incorrect answer! Only 'yes' and 'no' is allowed")                
                
            lat = psl_area.lat
            lon = psl_area.lon
            time = psl_area.time.values
            time_len = len(time)
            lon_list = list(lon)
            lat_list = list(lat)
            len_lon = len(lon_list)
            len_lat = len(lat_list)
            ds = []

            print('Calculating constants ☀︎') 

            lat_central = lat
            phi = lat_central            
            constants = JK_functions.constants(phi, lon)
            sc = constants[0]
            zwa = constants[1]
            zwb = constants[2]
            zsc = constants[3]            


            print('Checking time formats ☂︎')

                    #Checking the time coordinate values 
            if type(time[0]) == np.datetime64:
                time_pd = pd.to_datetime(time)
                dates = [pd.to_datetime(str(time_pd[t].year) + '-' + str(time_pd[t].month) + '-' + str(time_pd[t].day), 
                                            format = '%Y%m%d',errors = 'ignore') for t in range(len(time))]
            else:
                dates = [pd.to_datetime(str(time[t].year) + '-' + str(time[t].month) + '-' + str(time[t].day), 
                                    format = '%Y%m%d',errors = 'ignore') for t in range(len(time))]

                    #Extracting the 16-gridded values of MSLP
                    #extraction based on point 8 on original map
            print('Extracting 16 gridpoints ●')
            if answer_globe == 'yes':
                gridpoints = JK_functions.extracting_gridpoints_gcm_globe(mslp, lat, lon)
                p1  = gridpoints[0]
                p2  = gridpoints[1]
                p3  = gridpoints[2]
                p4  = gridpoints[3]
                p5  = gridpoints[4]
                p6  = gridpoints[5]
                p7  = gridpoints[6]
                p8  = gridpoints[7]
                p9  = gridpoints[8]
                p10 = gridpoints[9]
                p11 = gridpoints[10]
                p12 = gridpoints[11]
                p13 = gridpoints[12]
                p14 = gridpoints[13]
                p15 = gridpoints[14]
                p16 = gridpoints[15]
            elif answer_globe == 'no':
                gridpoints = JK_functions.extracting_gridpoints_gcm_area(mslp, lat, lon)
                p1  = gridpoints[0]
                p2  = gridpoints[1]
                p3  = gridpoints[2]
                p4  = gridpoints[3]
                p5  = gridpoints[4]
                p6  = gridpoints[5]
                p7  = gridpoints[6]
                p8  = gridpoints[7]
                p9  = gridpoints[8]
                p10 = gridpoints[9]
                p11 = gridpoints[10]
                p12 = gridpoints[11]
                p13 = gridpoints[12]
                p14 = gridpoints[13]
                p15 = gridpoints[14]
                p16 = gridpoints[15]
            else:
                raise TypeError("Incorrect answer! Only 'yes' and 'no' is allowed")                

            print('Computing flow terms ☈')
            #Computing equations of flows and vorticity                        
            flows = JK_functions.flows_gcm(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, sc, zwa, zsc, zwb, lat, lon, time)
            W  = flows[0] #Westerly flow
            S  = flows[1] #Southerly flow
            F  = flows[2] #Resultant flow
            ZW = flows[3] #Westerly shear vorticity
            ZS = flows[4] #Southerly shear vorticity
            Z  = flows[5] #Total shear vorticity            

        print('Computing flow directions ↖︎ → ↘︎ ↓ ←')
        #Computing the wind direction values          
        dd = np.arctan(W/S)
        deg = np.rad2deg(dd)
        deg=np.mod(180+np.rad2deg(np.arctan2(W, S)),360) 
        #https://confluence.ecmwf.int/pages/viewpage.action?pageId=133262398 
        #Assigning the Wind Direction labels
        direction = JK_functions.direction_def_NH(deg)
        if source == 'REAN':
            direction = xr.where(deg.latitude < 0, JK_functions.direction_def_SH(deg), direction)
        elif source == 'GCM':
            direction = xr.where(deg.lat < 0, JK_functions.direction_def_SH(deg), direction)            

        print('Determining the Circulation types ☁︎ ☀︎ ☂︎')
        #Determination of Circulation Type (27 Original types)
        lwt = JK_functions.assign_lwt(F, Z, direction)[0]
        
        #Storing the gridded Circulation Types in an xarray file
        print('Saving the data in an xarray format ✉︎')
        if (mslp.dims[1] == 'latitude') or (mslp.dims[1] == 'lat'):
            output=xr.DataArray(data = lwt.values,
                coords = {'time': time,
                        'lat': lat_list, 
                        'lon': lon_list},
                        dims = ['time', 'lat', 'lon'])
            output.name = 'CT' #Assigning variable name
        elif mslp.dims[1] == 'number':
            output=xr.DataArray(data = lwt.values,
                coords = {'time': time,
                          'number':mslp.number,
                          'lat': lat_list, 
                          'lon': lon_list},
                        dims = ['time','number', 'lat', 'lon'])
            output.name = 'CT' #Assigning variable name
        if source == 'GCM':
            output.attrs = {
                'description':'Gridded Lamb circulation types derived from MSLP data based on the automated Jenkinson-Collison classification',
                'institution_id': institution_id,
                'source_id': source_id,
                'experiment_id': experiment_id}
        elif source == 'REAN':
            output.attrs = {
                'description':'Gridded Lamb circulation types derived from MSLP data based on the automated Jenkinson-Collison classification'}
        order_lats = lat_list[0] - lat_list[-1]
        if order_lats < 0:
            output = output.reindex(lat=list(reversed(output.lat)))
        else:
            output = output        
        print('The End! ✓')
                # 'citation'


    else:
        raise TypeError("Incorrect filename directory. Only strings allowed")
    
    return output