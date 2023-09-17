"""
Script for estimating VWAP for futures contract.

The script takes as input the OHLC prices for the futures contracts, 
and computes the official prices as per CME methodology.

The script outputs a csv file with the official prices for each day.

"""

import numpy as np
import pandas as pd
import datetime
import cmeProjectSupportingFunctions as cpsf

# basic initialization
start = datetime.datetime(2022, 5, 1) 
end = datetime.datetime(2023, 6, 27) 

date_range = pd.date_range(start, end, freq='1min')
prices_df = pd.DataFrame(index=date_range)
volumes_df = pd.DataFrame(index=date_range)

all_possible_futures_names = cpsf.generate_futures_codes(years=['2', '3','4'])

# load futures' price data and estimate the VWAP for each minute
# note some details below are very specific about how we saved the data. The reader will likely need to change this
input_directory = '2022.05-2023.06///'
extension = '.csv'
for i in all_possible_futures_names:
    # import data
    try:   
        input_file = input_directory + i + extension
        data = pd.read_csv(input_file, index_col = 0)
        data.index = pd.to_datetime(data.index)
        print("CSV for future", i, "loaded")
    except:
        continue

    prices_df[i] = data['Close']
    volumes_df[i] = data['Volume']

# list of futures whose data is available
futures = prices_df.columns

# fill n/a values with 0 for volumes and the latest price for prices
# note this is only done _before_ the resampling, when presumably the 
# prices are n/a when the volumes happen to be 0 anyway
volumes_df = volumes_df.fillna(0)
prices_df = prices_df.fillna(method='ffill')
prices_df = prices_df.fillna(method='bfill')

# perform a resampling of the volumes for a specified window [30min as per CME]
resample_window = '30min'
resampled_volumes_df = volumes_df.resample(resample_window, label='right').sum()
resampled_prices_df = pd.DataFrame()

# perform a resampling of the prices
for future in futures: 
    resampled_prices_df[future] = ( (prices_df[future] * volumes_df[future]
            ).resample(resample_window, label='right').sum() / 
            resampled_volumes_df[future]
            )
    
for i, future in enumerate(resampled_prices_df.columns[:-1]):
    
    # when inspecing a future we just ffill and bfill its prices
    # so until we have a good print the value can be off but we know we have no NAN values
    resampled_prices_df[future] = resampled_prices_df[future].fillna(method='ffill')
    resampled_prices_df[future] = resampled_prices_df[future].fillna(method='bfill')
    
    # then we use it to populate the prices of the next future
    next_future = resampled_prices_df.columns[i+1]

    # we go though the time stamps one by one
    for j, time in enumerate(resampled_prices_df.index[1:]):
        price = resampled_prices_df.at[time, next_future]
        isPriceNAN = np.isnan(price)
        
        # what to do when a NaN price is found
        if isPriceNAN:
            prev_time = resampled_prices_df.index[j-1]
            prev_price_of_the_future_with_NAN_value = resampled_prices_df.at[prev_time, next_future] 
            
            # until we find the first valid future price we can just skip the step
            if np.isnan(prev_price_of_the_future_with_NAN_value):
                continue
            
            else:
                price_change_on_the_observable_future = resampled_prices_df.at[time, future] - resampled_prices_df.at[prev_time, future]
                predicted_price_on_NaN_future = prev_price_of_the_future_with_NAN_value + price_change_on_the_observable_future 
                resampled_prices_df.at[time, next_future] = predicted_price_on_NaN_future 

# identify the relevelant futures 
fut_specs = cpsf.generate_futures_specifications(years=['2', '3','4'])
full_day_by_day_date_range = pd.date_range(start, end, freq='1d')

# early close date for year 2022 and 2023
early_close_date = pd.to_datetime(['27-May-2022','01-Jul-2022','25-Nov-2022',
                                   '23-Dec-2022','30-Dec-2022','29-May-2023',
                                   '03-July-2023','24-Nov-2023','22-Dec-2023',
                                   '29-Dec-2023'])

# calculate the official futures price day by day
official_prices_df = pd.DataFrame(columns=futures)
for day in full_day_by_day_date_range:
    # identify the relevelant futures for that one day
    # 
    # CME weights the price observations by the volumes observed in ALL relevant futures
    # as opposed to by the volumes in the specific future being considered
    # we assume that the relevant futures are the 13 SER + 5 SFR that cover the 
    # period from that day till 1 year ahead
    relevant_futures = cpsf.identify_relevant_futures(day, fut_specs)
    
    # extract the volume data for the list of relevant futures and the specific day 
    extract_volumes_df = resampled_volumes_df[relevant_futures].sum(axis=1).copy()
    following_day = day + datetime.timedelta(1)
    extract_volumes_df = extract_volumes_df.loc[(extract_volumes_df.index >= day) & 
                                                (extract_volumes_df.index < following_day)]
    
    # extract the relevant prices data, so for the list of relevant futures and the specific day 
    extract_prices_df = resampled_prices_df.loc[extract_volumes_df.index, relevant_futures]
    
    # remove the data outside of the window of hours indicated by CME [7am-2pm]
    # for early close date, should 7am to 12pm
    in_window_volumes_df = extract_volumes_df.copy()
    in_window_volumes_df.index = in_window_volumes_df.index.time
    if day == pd.to_datetime('07-Apr-2023'):
        print('Early close for April 07, 2023 is special')
        in_window_volumes_df = in_window_volumes_df.loc[(in_window_volumes_df.index.astype(str) > '07:00:00') &
                                                        (in_window_volumes_df.index.astype(str) <= '10:00:00')]
    elif day in early_close_date:
        in_window_volumes_df = in_window_volumes_df.loc[(in_window_volumes_df.index.astype(str) > '07:00:00') &
                                                        (in_window_volumes_df.index.astype(str) <= '12:00:00')]
    else:
        in_window_volumes_df = in_window_volumes_df.loc[(in_window_volumes_df.index.astype(str) > '07:00:00') &
                                                        (in_window_volumes_df.index.astype(str) <= '14:00:00')]
    
    # if there are volumes for those futures on that day, 
    # then save their official prices
    if in_window_volumes_df.values.sum():
        in_window_prices_df = extract_prices_df.copy()   
        in_window_prices_df.index = in_window_prices_df.index.time
        in_window_prices_df = in_window_prices_df.loc[in_window_volumes_df.index]
        
        # re-scale the sum of all futures volumes for that day to 1
        rescaled_volumes_df = in_window_volumes_df / in_window_volumes_df.sum()
        rescaled_volumes_df = rescaled_volumes_df.values.reshape(-1, 1)
        
        # remove the n/a or empty prices
        in_window_prices_df = in_window_prices_df.dropna(axis=1)
        relevant_futures = in_window_prices_df.columns
        
        # calculate the official price as the price weighted by the volumes
        # of all futures in each time window as per CME methodogy
        # note this is not a pure VWAP
        prices_day = (in_window_prices_df * rescaled_volumes_df).sum()
        official_prices_df.loc[day, relevant_futures] = prices_day

# load the past daily o/n SOFR fixings 
SOFR_fixings = pd.read_csv('SOFR_FIXING_ALL.csv', index_col=0, dayfirst=False)
SOFR_fixings = SOFR_fixings[['SOFR']]
SOFR_fixings.index = pd.to_datetime(SOFR_fixings.index, dayfirst=True)

# create a DataFrame that contains the past SOFR fixings and the official prices
complete_output_df = pd.concat([SOFR_fixings, official_prices_df], axis = 1)
output_file_name = 'OFFICIALprices.csv'
complete_output_df.to_csv(output_file_name)
print("Official prices saved to file in", output_file_name)