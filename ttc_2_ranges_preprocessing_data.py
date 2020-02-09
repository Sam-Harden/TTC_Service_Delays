#!/usr/bin/env python
# coding: utf-8


######################## Import libraries, data sets ###############################

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

########################### Load TTC delay data files ###############################

# Load the .xlsx files contain TTC delay data from the 'delays_by_month' folder
# Print the list of file names, dimensions of DataFrames
print('Original Data:')
#print('Datatsets:')

files=[]
for f in os.listdir('1. Original Data/delays_by_month'):
    if f.endswith('.xlsx'):
        df_f = pd.read_excel('1. Original Data/delays_by_month/' + f)
        # Print file names and dimensionality of the DataFrames
        #print(f, df_f.shape) 
        files.append(df_f)

# Concatenate DataFrames to combine the data       
df_data = pd.concat(files)
print('\nСombined Data Set:', df_data.shape)
print(len(files),'files downloaded')


############### Load files with additional information (TTC delay codes) ################

# Load the .xls file with codes of delays
ttc_codes = '1. Original Data/ttc_data/Subway & SRT Log Codes.xls'
codes = pd.read_excel(ttc_codes)

# Transform two tables from 'Subway & SRT Log Codes.xls' file into a dataframe to combine all codes 
# Create two separate dataframes for SUB RMENU codes and SRT RMENU codes
codes_SUB = pd.DataFrame(codes[['SUB RMENU CODE', 'CODE DESCRIPTION']])
codes_SRT = pd.DataFrame(codes[['SRT RMENU CODE', 'CODE DESCRIPTION.1']])

# Drop empty rows created by Python from xls file. 
codes_SUB.dropna(inplace=True)
codes_SRT.dropna(inplace=True)

# Rename columns in both dataframes to 'Code' and 'Reason for delay'
codes_SUB.rename(columns={'SUB RMENU CODE':'Code', 'CODE DESCRIPTION':'Reason for delay'}, inplace=True)
codes_SRT.rename(columns={'SRT RMENU CODE':'Code', 'CODE DESCRIPTION.1':'Reason for delay'}, inplace=True)

# Concat two dataframes with codes into one 
codes_conc = pd.concat([codes_SUB,codes_SRT])

# Merging sabway delas data with code descriptions 
df = pd.merge(df_data, codes_conc, on='Code', how='left')
print(df.head())

##############################  Set data requirements for the project  #####################################

# Set minimum delay time, time, dates which will be considered in the project
time_delay = 15 # Delays 15 minutes and more
start_time_project = '06:00' # Delays after 6:00
start_date_project = '2016-10-01' # Delays since '2016-10-01' 
outliers_very_long_delays = 7 # A max number of 30 minutes units for too long delays.  
                              #Longer delays will be replaced by median for that type of delays  

#                                  AND
#
# Create two or three ranges of time delays in the 'Bins' section below  to get more consistent data sets: 
#        15-60 minutes and 60 minutes or more
#                                  OR
#        15-30 minutes, 30-60 minutes, and 60 minutes or more
 
    
############################################################################################################    
    
print('\nProject settings:')
print(f'Minimum delay time: {time_delay} minutes')
print(f'Delays after {start_time_project}')
print(f'Delays since {start_date_project} \n' )

# Filter the relevant data
# Filter the data by delay time
data = df[df['Min Delay'] >= time_delay]
# Filter delays since the specific date
start_date = pd.to_datetime(start_date_project)
data = data[data['Date']>start_date]
# Convert time from string format to datetime object with current date and save it into the 'Time_CurrentDate' column
data['Time_CurrentDate']=pd.to_datetime(data['Time'])
# Filter the data within specific time range
start_time = pd.to_datetime(start_time_project)
data = data[data['Time_CurrentDate']>start_time]
# Drop the 'Time_CurrentDate' column
data.drop('Time_CurrentDate', inplace=True, axis=1)

# Reset index
data = data.reset_index(drop=True)

# Drop columns which are not used in the project
drop_columns = ['Code','Min Gap','Vehicle']
data.drop(columns=drop_columns, axis=1, inplace=True)

print('Original Data Set:', df.shape)
print('Data Set for Project:', data.shape, '\n')

# Print a summary of the DataFrame 
# Get column dtypes, non-null values
print(data.info())

# Check duplicated rows
duplicated_rows = data[data.duplicated()].sort_values(by='Date')

# Drop duplicated rows
data.drop_duplicates(inplace=True)
data.duplicated().sum()

print('\nDuplicated rows: ', data.duplicated().sum())
print('Data set: ', data.shape)

# Reset index
data.reset_index(drop=True, inplace=True)

################################################# Bins  ##########################################################

#################### ------------------ The first option --------------------------------------------------------
# Create three ranges of time delays: 15-30 minutes, 30-60 minutes, and 60 minutes or more to get more consistent data sets
#bins = [15, 30, 60, (data['Min Delay'].max()+1)]

#################### ------------------ The second option -------------------------------------------------------
# Create two ranges of time delays: 15-60 minutes and 60 minutes or more to get more consistent data sets
bins = [15, 60, (data['Min Delay'].max()+1)]

###################################################################################################################

print('\nProject settings:')
print('Bins: ', bins, '\n')

# Save duplicates rows to the 'Analysis' folder 
# If 3 time ranges were created
if len(bins)==4:
    duplicated = '2. Analysis/ttc_3_ranges_duplicated_rows.xlsx'
# If 2 time ranges were created
elif len(bins)==3:
    duplicated = '2. Analysis/ttc_2_ranges_duplicated_rows.xlsx'

duplicated_rows.to_excel(duplicated)

data['Delay_Time_Range'] = pd.cut(data['Min Delay'], bins = bins, right=False)
data['Delay_Time_Range'] = data['Delay_Time_Range'].astype(str)

print(data['Delay_Time_Range'].value_counts())

# Create a list of time ranges for hue order on plots and later references
time_ranges_list=data['Delay_Time_Range'].unique().tolist()
print(time_ranges_list) 

# Create three dataframes based on the the time ranges
delays_small = data[(data['Min Delay'] >= bins[0]) & (data['Min Delay'] < bins[1])]
delays_medium = data[(data['Min Delay'] >= bins[1]) & (data['Min Delay'] < bins[2])]
delays_large = data[data['Min Delay'] >= bins[2]]

# Create a list of dataframes for later references
list_df = [delays_small, delays_medium, delays_large]


####################################### Common settings #################################################

# Legends, ticks for plots, dataframes
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
weekdays = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
hours = ['05:30','06:00','06:30','07:00','07:30','08:00','08:30','09:00','09:30','10:00','10:30','11:00','11:30','12:00','12:30','13:00','13:30','14:00','14:30','15:00','15:30','16:00','16:30','17:00','17:30','18:00','18:30','19:00','19:30','20:00','20:30','21:00','21:30','22:00','22:30','23:00','23:30','00:00','00:30','01:00','01:30','02:00','02:30','03:00','03:30','04:00','04:30','05:00']

# Group data in 'Bound' and count number of bounds by line 
bounds = data.groupby(['Line','Bound'])[['Bound']].count()
bounds_sum = bounds.sum()
#print(bounds)
#print('Total:\n', bounds_sum)

# Select entries with mismatch between a line and a bound by line
df_BD = data[(data['Line']=='BD') & ~((data['Bound']=='E') | (data['Bound']=='W'))].sort_values(by='Station')
df_SHP = data[(data['Line']=='SHP') & ~((data['Bound']=='E') | (data['Bound']=='W'))].sort_values(by='Station')
df_SRT = data[(data['Line']=='SRT') & ~((data['Bound']=='N') | (data['Bound']=='S'))].sort_values(by='Station')
df_YU = data[(data['Line']=='YU') & ~((data['Bound']=='N') | (data['Bound']=='S'))].sort_values(by='Station')
# Concatenate DataFrames 
df_bounds = pd.concat([df_BD, df_SHP, df_SRT, df_YU])
# Select rows where 'Bound' is not missing
df_bounds = df_bounds[~df_bounds['Bound'].isnull()]
# Save entries with mismatch between a line and a bound to the 'Analysis' folder
# If 3 time ranges were created
if len(bins)==4:
    line_bound = '2. Analysis/ttc_3_ranges_bound_line_missmatch_rows.xlsx'
# If 2 time ranges were created
elif len(bins)==3:
    line_bound = '2. Analysis/ttc_2_ranges_bound_line_missmatch_rows.xlsx'
df_bounds.to_excel(line_bound)
    
# Correct a line or a bound
# BD line
# Replace incorrect line 'BD' with 'YU'
BD_to_YU = (data['Line']=='BD') & ((data['Station']=='ST CLAIR WEST STATION') | (data['Station']=='YORKDALE STATION')| (data['Station']=='ST PATRICK STATION'))
data['Line']=data['Line'].mask(BD_to_YU, 'YU')
data['Bound'] = data['Bound'].replace({'R':'E'})

# SRT line
# Replace 'E' >> 'N'
SRT_E_N = (data['Line']=='SRT') & (data['Bound']=='E')
data['Bound']=data['Bound'].mask(SRT_E_N, 'N')
# Replace 'W' >> 'S'
SRT_W_S = (data['Line']=='SRT') & (data['Bound']=='W')
data['Bound']=data['Bound'].mask(SRT_W_S, 'S')
# Replace '66' line >> SRT
data['Line'] = data['Line'].replace({'66': 'SRT'})

# YU line
# Replace '5' >> 'S'
data['Bound'] = data['Bound'].replace({'5':'S'})
# Replace incorrect line 'YU' with 'BD'
YU_to_BD = (data['Line']=='YU') & ((data['Station']=='YONGE BD STATION') | (data['Station']=='KENNEDY BD STATION') )
data['Line']=data['Line'].mask(YU_to_BD, 'BD')

# Check coordination of a line and a bound 
bounds_res = data.groupby(['Line','Bound'])[['Bound']].count()
bounds_res_sum = bounds_res.sum()

# Check missing values
missing_bounds = data[data['Bound'].isnull()].sort_values(by='Line')
# Save the rows with  missing bound to the 'Analysis' folder
# If 3 time ranges were created
if len(bins)==4:
    bounds = '2. Analysis/ttc_3_ranges_missing_bound_rows.xlsx'
# If 2 time ranges were created
elif len(bins)==3:
    bounds = '2. Analysis/ttc_2_ranges_missing_bound_rows.xlsx'
missing_bounds.to_excel(bounds)

# Found mode of 'Bound' by line
mode_bound_BD = data[(data['Line']=='BD')]['Bound'].mode()[0]
mode_bound_SHP = data[(data['Line']=='SHP')]['Bound'].mode()[0]
mode_bound_SRT = data[(data['Line']=='SRT')]['Bound'].mode()[0]
mode_bound_YU = data[(data['Line']=='YU')]['Bound'].mode()[0]

# Fill missing 'Bound' values with mode
bound_BD_null = (data['Line']=='BD') & (data['Bound'].isnull())
data['Bound']=data['Bound'].mask(bound_BD_null, mode_bound_BD)

bound_SHP_null = (data['Line']=='SHP') & (data['Bound'].isnull())
data['Bound']=data['Bound'].mask(bound_SHP_null, mode_bound_SHP)

bound_SRT_null = (data['Line']=='SRT') & (data['Bound'].isnull())
data['Bound']=data['Bound'].mask(bound_SRT_null, mode_bound_SRT)

bound_YU_null = (data['Line']=='YU') & (data['Bound'].isnull())
data['Bound']=data['Bound'].mask(bound_YU_null, mode_bound_YU)

# Check missing values are filled
# 'Reason for delay' column will be dropped later 
data.drop('Reason for delay', axis=1, inplace=True)
print('\nMissing values:') 
print(data.isnull().sum())

#########################################  Feature Engineering  ##############################################

# Extract month from datetime object
data['Month']=data['Date'].dt.month
# Extract weekday from datetime object
data['Weekday']=data['Date'].dt.weekday

# Create DateTime object using DateTime object from the 'Date' column and time in string format from the 'Time' column
data['Date_Time']=pd.to_datetime(data['Date'].astype(str) + ' ' + data['Time'])
# Calculate end of delay time by adding 'Min Delay' value as timeTime_Delta 
data['End_Delay_Time']=data['Date_Time']+data['Min Delay'].apply(lambda x:timedelta(minutes=x))

# Round time to 30 minutes from 'Date_Time' DateTime object and save it to the new column 'Start_Time_Round_30_min'
data['Start_Time_Round_30_min']=data['Date_Time'].dt.round('30min')
# Round  end of delay time to 30 minutes from 'End_Delay_Time' DateTime object and save it to the new column 'End_Time_Round_30_min'
data['End_Time_Round_30_min']=data['End_Delay_Time'].dt.round('30min')
# Count a number of 30 minutes units in a delay time
data['Time_Delta']=(data['Min Delay']/30).round().astype(int)
# Set 1 if unit is 0 as delay time was less than 30 minutes 
data['Time_Delta']=data['Time_Delta'].apply(lambda x: 1 if x==0 else x)

# Create lists of time ranges  
data['Range']=data.apply(lambda x: pd.date_range(start=x['Start_Time_Round_30_min'], periods=x['Time_Delta'], freq='30min').time.tolist(), axis=1)
# Copy indexes into the 'Row_Index' column
data['Row_Index']=data.index

# Create mask for outlier of very long delays
#outliers_very_long_delays = 7
mask_time_delta = data['Time_Delta'] > outliers_very_long_delays
# Get minimum of 'Time_Delta' for outlier of very long delays
median_time_delta = data[mask_time_delta]['Time_Delta'].min()
# Replace 'Time_Delta' values of very long delay with median
data['Time_Delta'] = data['Time_Delta'].mask(mask_time_delta, median_time_delta)

def get_time_string(list_time):
    """ Converts a list of datetime.time objects into a list of strings in '%H:%M' format. """
    # Creates a list
    list_time_str=[]
    # Loop through the list 
    for time in list_time:
        # Append a string converted in '%H:%M' format into the list
        list_time_str.append(time.strftime('%H:%M'))
    
    return list_time_str

# Convert lists of datetime.time objects into lists of strings in '%H:%M' format
data['Range_List']=data['Range'].apply(lambda x: get_time_string(x))

# Create DataFrames from lists of time ranges
data['Time_DF'] = data.apply(lambda x: pd.DataFrame(x['Range_List'], index=x['Range_List'], columns=[x['Row_Index']]).transpose(), axis=1)

# Create a DataFrame with  columns equals hours
df_hours=pd.DataFrame(columns=hours)

# Concatenate the DataFrame of hours grid and DataFrames with time ranges
for i in range(0,len(data)):
    df_hours = pd.concat([df_hours,data['Time_DF'][i]], sort=False, axis=0)

#print('\nDataFrame of time ranges:', df_hours.shape)

# Replace non-zero values in the DataFrame with 1
df_hours = df_hours.notnull().astype(int)

# Join the main and  the united time ranges grid DataFrames
data = data.join(df_hours, how='outer')

# Round time to 30 minutes from 'Start_Time_Round_30_min' DateTime object and save it to the new column 'Round_Time_30_min' as string in '%H:%M' format
data['Round_Time_30_min']=data['Start_Time_Round_30_min'].apply(lambda x: x.strftime('%H:%M'))
# Create a DataFrame by filter data to use for plotting delays by 30 minutes periods 
data_time=data[['Round_Time_30_min','Delay_Time_Range']]

# Convert 'Date' to string format and save to the 'Holiday' column
data['Holiday']=data['Date'].astype(str)
# Create a list of holidays in Ontario in 2018 and 2019
ontario_holidays = ['2018-01-01', '2018-02-19', '2018-03-30', '2018-05-21', '2018-07-02', '2018-09-03', '2018-10-08', '2018-12-25', '2018-12-26', '2019-01-01', '2019-02-18', '2019-04-19', '2019-05-20', '2019-07-01', '2019-09-02', '2019-10-14', '2019-12-25', '2019-12-26'] 
# Applay filter to convert holidays to 1 and a weekday to 0
data['Holiday']=data['Holiday'].apply(lambda x: 1 if x in ontario_holidays else 0)
#data['Holiday'].value_counts()

# Extract 'Saturday' and 'Sunday' to the 'Weekend' column
data['Weekend']=data['Weekday'].apply(lambda x: 1 if ((x==5) | (x==6)) else 0)

# Create column for weekend and holidays
data['Weekend_Holiday']=data['Weekend'] + data['Holiday']

# Create new 'Line_Bound'column
data['Line_Bound']=data['Line']+' '+data['Bound']

# Drop columns which were used for feature engineering and calculations
#drop_columns = ['Date', 'Time', 'Day', 'Station', 'Min Delay', 'Bound', 'Line','Reason for delay', 'Date_Time',

drop_columns = ['Date', 'Time', 'Day', 'Station', 'Min Delay', 'Bound', 'Line','Date_Time',
       'End_Delay_Time', 'Start_Time_Round_30_min', 'End_Time_Round_30_min',
       'Time_Delta', 'Range', 'Row_Index', 'Range_List', 'Time_DF','Round_Time_30_min', 'Weekend','Holiday']
data.drop(drop_columns, axis=1, inplace=True)

# Create a list
drop_null_columns = []
# Loop through columns
for data_column in data.columns:
    # Add time range columns with only 0 values to the list
    if data[data_column].sum()==0:
        drop_null_columns.append(data_column)

#print(f'Drop empty time range columns {drop_null_columns}')

# Drop empty time range columns 
data.drop(drop_null_columns, axis=1, inplace=True)

# Convert categorical variable into dummy/indicator variables
cat_features =['Month', 'Weekday', 'Line_Bound']
data = pd.get_dummies(data, columns=cat_features, drop_first=True)

print(f'\nPrepared Data: {data.shape}')

# Save the data to the 'Prepared Data' folder
# If 3 time ranges were created
if len(bins)==4:
    prepared_data = '3. Prepared Data/ttc_3_ranges_prepared_data.xlsx'
# If 2 time ranges were created
elif len(bins)==3:
    prepared_data = '3. Prepared Data/ttc_2_ranges_prepared_data.xlsx'

data.to_excel(prepared_data, index=False)
print(f"The file '{prepared_data}' is saved.")


