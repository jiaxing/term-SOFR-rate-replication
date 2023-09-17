"""
Script for modelling forward SOFR curve given a set of SOFR futures prices, and 
estimating the equivalent Term SOFR rates
Section 3 and 4 of the Paper: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4566882

The script uses the official prices already stored in csv file 'OFFICIALprices.csv'.
The data cannot be shared so the reader will need to download it independently. 
It can rely on the VWAP of SOFR futures achieved by any hedging strategy. 
"""

# import needed libraries
import numpy as np
import pandas as pd
import datetime
import cmeProjectSupportingFunctions as cpsf
from scipy.optimize import minimize


# load the futures prices
file_of_futures_levels = 'OFFICIALprices.csv'
market_data_df = pd.read_csv(file_of_futures_levels, index_col=0, dayfirst=True)
market_data_df.index = pd.to_datetime(market_data_df.index, dayfirst=True)

# load the futuers daily volumes data
market_volumes_df = pd.read_csv('OFFICIALvolumes.csv', index_col=0)
market_volumes_df.index = pd.to_datetime(market_volumes_df.index, dayfirst=True)

# Define the FOMC dates, which are the dates after the FOMC meetings
# See details in : https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
FOMC_dates = pd.to_datetime([
        "16-Dec-21", "27-Jan-22", "17-Mar-22", "05-May-22", "16-Jun-22", 
        "28-Jul-22", "22-Sep-22", "03-Nov-22", "15-Dec-22", "02-Feb-23", 
        "23-Mar-23", "04-May-23", "15-Jun-23", "27-Jul-23", "21-Sep-23", 
        "02-Nov-23", "14-Dec-23", '01-Feb-24', '21-Mar-24', '02-May-24', 
        '13-Jun-24', '01-Aug-24', '19-Sep-24', '08-Nov-24', '19-Dec-24'])

# define the holidays to be excluded from the date ranges
# Holidays info from https://www.sifma.org/resources/general/us-holiday-archive/
global EXCLUDED_HOLIDAYS
EXCLUDED_HOLIDAYS = pd.to_datetime([
    '17-Jan-22', '21-Feb-22', '15-Apr-22','30-May-22', '20-Jun-22', '4-Jul-22',
    '05-Sep-22', '10-Oct-22', '11-Nov-22','24-Nov-22', '26-Dec-22', '02-Jan-23',
    '16-Jan-23', '20-Feb-23', '29-May-23','19-Jun-23', '04-Jul-23', '04-Sep-23',
    '09-Oct-23', '23-Nov-23', '25-Dec-23','01-Jan-24', '15-Jan-24', '19-Feb-24',
    '29-Mar-24', '27-May-24', '19-Jun-24','04-Jul-24', '02-Sep-24', '14-Oct-24',
    '11-Nov-24', '28-Nov-24', '25-Dec-24','01-Jan-25'])

# initialise a df that contains the dates and jumps in SOFR rates
FOMC_vector = pd.DataFrame(data=0, index=[date for date in FOMC_dates], columns=['Jumps'])

# initialize all bussiness day 
business_date = pd.bdate_range(start=pd.to_datetime('02-May-22'), end=pd.to_datetime('01-Jul-23'), freq='C', holidays=EXCLUDED_HOLIDAYS)

# drop holidays from market data
market_data_df = market_data_df.drop([day for day in EXCLUDED_HOLIDAYS if day in market_data_df.index])

# get future contracts info
fut_specs_df = cpsf.generate_futures_specifications(['2', '3', '4'])

# 
TERM_SOFR_tenors_in_months = [1, 3, 6, 12]
TERM_SOFR_estimates = pd.DataFrame(index=market_data_df.index, 
                                    columns=TERM_SOFR_tenors_in_months)


def identify_relevant_FOMC_dates(day, FOMC_vector, fut_specs_df, relevant_futures):
    """
    Function to identify the relevant FOMC dates for a given day, as per CME Group's methodology

    Parameters
    ----------
    day : datetime
        The estimation date.
    FOMC_vector : pd.DataFrame
        The vector of FOMC dates and jumps.
    fut_specs_df : pd.DataFrame
        The specifications of the futures contracts.
    relevant_futures : list
        The list of relevant futures contracts.

    Returns
    -------
    relevant_FOMC_dates : list
        The list of relevant FOMC dates.
    """

    minimum_date = day
    maximum_date = max(fut_specs_df.loc[relevant_futures]['End'])
    relevant_FOMC_dates = [date for date in FOMC_vector.index
                           if date >= minimum_date and date < maximum_date]
    return relevant_FOMC_dates


# build SOFR by using the past SOFR prints and FOMC jumps
def build_fitted_curve(day, market_data_df, Theta_vector, relevant_FOMC_dates):
    """
    Function for building the fitted curve for a specific day.

    Parameters
    ----------
    day : datetime
        The estimation date.
    market_data_df : pd.DataFrame
        The DataFrame containing the market price data.
    Theta_vector : pd.DataFrame
        The DataFrame containing the FOMC dates and jumps.
    relevant_FOMC_dates : list of datetime
        The list of FOMC dates relevant for the specific day.

    Returns
    -------
    curve_df : pd.DataFrame
        The DataFrame containing the fitted curve for SOFR rates.
    """    
    # create the date ranges to analyse
    lookback_horizon = 100
    lookforward_horizon = 15 * 30 + 9

    start_date = day - datetime.timedelta(lookback_horizon)
    prev_day = day - datetime.timedelta(1)
    end_date = day + datetime.timedelta(lookforward_horizon)

    # generate the ranges
    past_date_range = pd.bdate_range(start=start_date, end=prev_day, freq='C', holidays=EXCLUDED_HOLIDAYS)
    fwd_date_range = pd.bdate_range(start=day, end=end_date, freq='C', holidays=EXCLUDED_HOLIDAYS)

    # populate past_curve_df 
    past_curve_df = market_data_df.loc[past_date_range, ['SOFR']].copy()
    
    # pre-populate fwd_curve_df 
    fwd_curve_df = pd.DataFrame(data=0, index=fwd_date_range, columns=['SOFR'])

    if fwd_date_range[0] not in relevant_FOMC_dates:
        stub = Theta_vector.values[0]
        fwd_curve_df += stub

    for FOMC_date in relevant_FOMC_dates:
        jump = Theta_vector.loc[FOMC_date].values[0]
        fwd_curve_df.loc[fwd_curve_df.index >= FOMC_date] += jump 

    # merge the df and create one whole curve df
    curve_df = pd.concat([past_curve_df, fwd_curve_df], axis=0)
    
    return curve_df


def calculate_implied_futures_price(curve, calendar_curve, start, end, fut_type):
    """
    Function for calculating the implied futures price from the fitted curve.

    Parameters
    ----------
    curve : pd.DataFrame
        Dataframe that contains the column 'SOFR' which indicates the rate
        SOFR rate for any business day
    start : datetime
        Swap start date for the future
    end : datetime
        Swap end date for the future
    fut_type : string
        It can either be 'm' for monthly future or 'q' for quarterly

    Returns
    -------
    price : float
        Implied futures price from the curve
        Specifications as per CME were used: https://www.cmegroup.com/education/files/sofr-futures-contract-specifications.pdf
    """ 
    rate = 0.0 if fut_type == 'm' else 1
    
    if fut_type == 'm':
        # get the end date of montly futures
        end = start + pd.DateOffset(months=1) - pd.DateOffset(days=1)
        # get the calendar days sofr curve at specific month and calculate the average rate
        rate = calendar_curve.loc[start:end].mean()[0]

    # if it is a quarterly future, annualise the rate 
    if fut_type == 'q': 
        period_actual_days = (end - start).days
        curve = curve.loc[(curve.index >= start) & (curve.index <= end)].copy()
        for idx in np.arange(0, len(curve) - 1):
            daily_rate = curve.iloc[[idx]].values[0][0]
            if curve.index[idx+1] <= end:
                days_of_validity = (curve.iloc[[idx+1]].index - curve.iloc[[idx]].index).days[0]
            else:
                days_of_validity = (end - curve.iloc[[idx]].index).days[0]

            rate *= 1 + daily_rate / 100 * days_of_validity / 360
        rate = (rate - 1) * 100 * 360 / period_actual_days 
        
    price = 100 - rate
    return price


def price_the_whole_futures_strip(curve_df, calendar_curve, fut_specs_df, relevant_futures):  
    """
    Function for pricing the whole futures strip, as a loop of function calculate_implied_futures_price

    Parameters
    ----------
    curve_df : pd.DataFrame
        Dataframe that contains the column 'SOFR' which indicates the SOFR
        for any business day
    calendar_curve : pd.DataFrame
        Dataframe that contains the column 'SOFR' which indicates the rate
        for each calendar day
    fut_specs_df : pd.DataFrame
        The DataFrame containing the specifications of all the futures.
    relevant_futures : list of strings
        The list of futures relevant for the specific day.

    Returns
    -------
    implied_futures_prices : pd.DataFrame
        The DataFrame containing the implied futures prices.
    """         
    # create a df to store the implied futures prices
    implied_futures_prices = pd.DataFrame(index=relevant_futures, columns=['Price'])    

    # calculate the implied futures prices
    for future in implied_futures_prices.index:
        start = fut_specs_df.loc[future, 'Start']
        end = fut_specs_df.loc[future, 'End']
        fut_type = fut_specs_df.loc[future, 'Type']
        price = calculate_implied_futures_price(curve_df, calendar_curve, start, end, fut_type)
        implied_futures_prices.loc[future, 'Price'] = price
    
    return implied_futures_prices


def generate_weights_for_futures_prices(day, futures_prices, futures_volume, volume_weight = False):
    """   
    Function for generating weights for futures. Used in cost function computation.
    Default method is equal-weight. Another one is volume-weight.

    Parameters
    ----------
    day : datetime
        The estimation date.
    futures_prices : pd.DataFrame
        The DataFrame containing the futures prices.
    futures_volume : pd.DataFrame
        The DataFrame containing the futures volumes.
    volume_weight : bool, optional
        Whether to use volume-weight. The default is False.

    Returns
    -------
    weights : pd.DataFrame
        The DataFrame containing the weights for futures.
    """
    if volume_weight:
        # get volume data on 'day' for all relevant futures
        volume_relevant = futures_volume.loc[day, futures_prices.index]
        total_volumes_on_day = volume_relevant.values.sum()
        # compute the volume-based weight for each relevant future
        weight_coeff = [i / total_volumes_on_day for i in volume_relevant.values]

    else:
        weight_coeff = 1 / len(futures_prices) # equal weight 
    
    weights = pd.DataFrame(data=weight_coeff, index=futures_prices.index, columns=['Weights'])

    return weights 


def estimate_TERM_SOFR_fixing(day, curve_df, calendar_curve, TERM_SOFR_tenors_in_months):
    """
    Function for computing the TERM SOFR fixing for a specific day given a fitted curve

    Parameters
    ----------
    day : datetime
        The estimation date.
    curve_df : pd.DataFrame
        Dataframe that contains the column 'SOFR' which indicates the SOFR
        for any business day
    calendar_curve : pd.DataFrame
        Dataframe that contains the column 'SOFR' which indicates the rate
        for each calendar day

    Returns
    -------
    estimates : list of floats
        The list of estimated TERM SOFR fixings.
    """
    estimates = []

    # the term starts from t+3 till the end of the tenor 
    day_idx = list(curve_df.index).index(day)
    start = curve_df.index[day_idx + 3]

    for tenor in TERM_SOFR_tenors_in_months:
        end = cpsf.add_time(start, months = tenor)
        end = cpsf.modfol(end, curve_df.index)
        
        # price the SOFR swap
        # note that the formula is equivalent as the quarterly SOFR future
        rate = 100 - calculate_implied_futures_price(curve_df, calendar_curve, start, end, fut_type='q')
        
        # CME states that the fixing can have up to 5 decimals
        rate = np.round(rate, 5)
        estimates.append(rate)
        
    return estimates


def cost_function(day, market_data_df, implied_futures_prices, market_volumes_df, 
                  Theta_vector, relevant_FOMC_dates,
                  method='MSE'):
    """
    Cost function used to calibrate the SOFR curve 

    Parameters
    ----------
    market_data_df : pd.DataFrame
        The DataFrame containing the market price data. 
    implied_futures_prices : pd.DataFrame
        The DataFrame containing the implied futures prices.
    FOMC_vector : pd.DataFrame
        The DataFrame containing the FOMC dates and jumps.
    market_volumes_df : pd.DataFrame
        The DataFrame containing the futures volumes.
    Theta_vector : pd.DataFrame
        The DataFrame containing the FOMC dates and jumps.
    relevant_FOMC_dates : list of datetime
        The list of FOMC dates relevant for the specific day.
    method : string, optional
        The method to be used. The default is 'MSE', ie Mean Squared Errors.
    
    Returns
    -------
    total_cost : Float
        The total cost of the cost function.
    """
    weights = generate_weights_for_futures_prices(day, implied_futures_prices, market_volumes_df, volume_weight=False)

    
    cumul_error = 0
    if method == 'MSE':
        # this approach is consistent with CME Group and Heitfield and Park (2019) 
        for future in implied_futures_prices.index:
            error = (market_data_df.loc[day, future] - implied_futures_prices.loc[future, 'Price']) ** 2
            error *= weights.loc[future].values[0]
            cumul_error += error
        cumul_error = np.sqrt(cumul_error)
        lambda_ = 0.1 / np.sqrt(len(relevant_FOMC_dates))

    elif method == 'MAE':
        # this approach calculates the mean absolute errors, which is in our view more consistent with industry practice
        for future in implied_futures_prices.index:
            error = (market_data_df.loc[day, future] - implied_futures_prices.loc[future, 'Price'])
            error = np.abs(error)
            error *= weights.loc[future].values[0]
            cumul_error += error
        lambda_ = 0

    # this is the L2 regularisation term, where all jumps are summed up
    # theta_0 should not be penalized 
    if lambda_: 
        Theta_k = Theta_vector.loc[relevant_FOMC_dates]
        regularisation_cost = sum([jump ** 2 for jump in Theta_k.values])
        regularisation_cost = np.sqrt(regularisation_cost)
        regularisation_cost *= lambda_
        cumul_error += regularisation_cost
    
    cumul_error = cumul_error[0]
    return cumul_error 


def calculate_cost(x, day, market_data_df, relevant_futures, idx, market_volumes_df, relevant_FOMC_dates, method='MSE'):
    """
    Function for calculating the cost of a specific curve configuration, to be run during optimisation.

    Parameters
    ----------
    x : numpy array
        The array containing the jumps.
    day : datetime
        The estimation date.
    market_data_df : pd.DataFrame
        The DataFrame containing the market price data.
    relevant_futures : list of strings
        The list of futures relevant for the specific day.
    idx : list of datetime
        The list of FOMC dates relevant for the specific day.
    market_volumes_df : pd.DataFrame
        The DataFrame containing the futures volumes.
    relevant_FOMC_dates : list of datetime
        The list of FOMC dates relevant for the specific day.
    method : string, optional
        The method to be used. The default is 'MSE', ie Mean Squared Errors.

    Returns
    -------
    cost : float
        The cost function value.
    """
    # transform numpy to DataFrame
    Theta_vector = pd.DataFrame(x, index=idx, columns=['Jumps'])

    # compute implied prices
    curve_df = build_fitted_curve(day, market_data_df, Theta_vector, relevant_FOMC_dates)
    
    # calendar curve for monthly futures pricing
    all_calendar_days = pd.date_range(curve_df.index[0], curve_df.index[-1])
    calendar_curve = pd.DataFrame(index=all_calendar_days, columns=['SOFR'])
    calendar_curve.loc[curve_df.index] = curve_df
    calendar_curve = calendar_curve.fillna(method='ffill')

    # could put fut_specs_df outside and call it anytime 
    fut_specs_df = cpsf.generate_futures_specifications(['2','3','4'])
    implied_futures_prices = price_the_whole_futures_strip(curve_df, calendar_curve, fut_specs_df, relevant_futures)

    # compute cost function
    cost = cost_function(day, market_data_df, implied_futures_prices, market_volumes_df, Theta_vector, relevant_FOMC_dates, method)
    return cost


def run_optimisation(day, market_data_df, Theta_vector, relevant_futures, market_volumes_df, relevant_FOMC_dates, method='MSE'):   
    """
    Function for running the optimisation.

    Parameters
    ----------
    day : datetime
        The estimation date.
    market_data_df : pd.DataFrame
        The DataFrame containing the market price data.
    Theta_vector : pd.DataFrame
        The DataFrame containing the FOMC dates and jumps.
    relevant_futures : list of strings
        The list of futures relevant for the specific day.
    market_volumes_df : pd.DataFrame
        The DataFrame containing the futures volumes.
    relevant_FOMC_dates : list of datetime
        The list of FOMC dates relevant for the specific day.
    initial_df : pd.DataFrame
        The DataFrame containing the initial guess of the Theta vector.
    method : string, optional
        The method to be used. The default is 'MSE', ie Mean Squared Errors.

    Returns
    -------
    result.x : numpy array
        The array containing the jumps.
    cost : float
        The cost function value.
    """    
    print("Running optimisation for day", day)
    idx = Theta_vector.index
    initial_guess = Theta_vector.to_numpy().reshape(len(Theta_vector))
    result = minimize(calculate_cost, initial_guess, method='BFGS', 
                      args=(day, market_data_df, relevant_futures, idx, market_volumes_df, relevant_FOMC_dates, method), 
                      options={'disp': True, 'return_all': True})
    print(result.message)
    cost = result.fun
    return result.x, cost



# =============================================================================
# Main Script
# =============================================================================

# set start date and end date for estimation task
start_date_of_the_calculations, end_date_of_the_calculations = '2022-05-02', '2023-06-27'

history_of_costs = pd.DataFrame(index=market_data_df.index, columns=['cost'])
prices_error = pd.DataFrame()
Theta_vector_all = pd.DataFrame()
for day in market_data_df.loc[start_date_of_the_calculations : end_date_of_the_calculations].index:    
    
    day = pd.to_datetime(day)
    # if there are not futures prices, just skip the day
    if not market_data_df.loc[day].drop(['SOFR']).any(): 
        continue

    # create the Theta_vector by using only the relevant jumps
    relevant_futures = cpsf.identify_relevant_futures(day, fut_specs_df)
    relevant_FOMC_dates = identify_relevant_FOMC_dates(day, FOMC_vector, fut_specs_df, relevant_futures)
    Theta_vector = FOMC_vector.loc[relevant_FOMC_dates].copy()

    # add a theta_0 element as stub, if needed 
    if day not in relevant_FOMC_dates:
        theta_0 = pd.DataFrame(data=0, index=[day], columns = ['Jumps']) 
        Theta_vector = pd.concat([theta_0, Theta_vector])
        
    # only optimise using the futures that have an official price and volume is not 0
    futures_with_price = list(market_data_df.loc[[day]].dropna(axis=1).columns[1:]) 
    day_volume_frame = market_volumes_df.loc[[day], futures_with_price]
    futures_with_data = list(day_volume_frame.columns[(day_volume_frame != 0).any()])
    # removes the futures name for which we have no data 
    relevant_futures = list(set(relevant_futures) & set(futures_with_data))

    # run optimization
    # choose method 'MSE' for the calibration consistent with CME Group's approach, 
    # or 'MAE' for mean absolute errors
    method = 'MSE'
    forward_result, cost = run_optimisation(day, market_data_df, Theta_vector, relevant_futures, market_volumes_df, relevant_FOMC_dates, method)
    Theta_vector = pd.DataFrame(forward_result, index=Theta_vector.index, columns = ['Jumps'])
    
    # record the optimal theta vector for each estimation
    Theta_record = pd.DataFrame(forward_result, index=Theta_vector.index, columns = [day])
    Theta_vector_all = pd.concat([Theta_vector_all, Theta_record], axis = 1)

    # update the best guess of the FOMC_vector using the latest optimum Theta_vector 
    FOMC_vector.loc[relevant_FOMC_dates] = Theta_vector.loc[relevant_FOMC_dates]

    # construct a fitted curve with optimised parameters
    curve_df = build_fitted_curve(day, market_data_df, Theta_vector, relevant_FOMC_dates)
    
    # calendar curve for monthly futures pricing
    all_calendar_days = pd.date_range(curve_df.index[0], curve_df.index[-1])
    calendar_curve = pd.DataFrame(index=all_calendar_days, columns=['SOFR'])
    calendar_curve.loc[curve_df.index] = curve_df
    calendar_curve = calendar_curve.fillna(method='ffill')

    # get difference of implied prices and actual prices after optimization
    estimated_prices = price_the_whole_futures_strip(curve_df, calendar_curve, fut_specs_df, relevant_futures)
    actual_prices = market_data_df.loc[[day], estimated_prices.index]
    diff_prices = actual_prices.T.to_numpy() - estimated_prices.to_numpy()
    diff_prices = pd.DataFrame(diff_prices, index = estimated_prices.index, columns = [day])
    prices_error = pd.concat([prices_error, diff_prices], axis = 1)

    # recored cost functionv value for each estimation
    history_of_costs.loc[day] = cost 

    estimates = estimate_TERM_SOFR_fixing(day, curve_df, calendar_curve, TERM_SOFR_tenors_in_months)
    TERM_SOFR_estimates.loc[day] = estimates
    print('\nEstimated TERM SOFR fixings for day', day, ':', estimates)


# Compare estimated results with the real fixings
estimate_term_sofr = TERM_SOFR_estimates.dropna().shift(1).dropna()

# import actual cme term sofr data
real_term_sofr = pd.read_csv('SOFR_FIXING_ALL.csv', index_col=0)
real_term_sofr.index = pd.to_datetime(real_term_sofr.index, dayfirst=True)
Term_SOFR = real_term_sofr[['1','3','6','12']]

# compute the error between estimated term sofr and actual term sofr
actual_term_sofr = Term_SOFR.loc[estimate_term_sofr.index]
difference = actual_term_sofr.to_numpy() - estimate_term_sofr.to_numpy()
diff_frame = pd.DataFrame(difference)
diff_frame.columns = actual_term_sofr.columns
diff_frame.index = actual_term_sofr.index
diff_frame = diff_frame.apply(pd.to_numeric)

# output results into excel files
diff_frame.to_excel('diff.xlsx')
history_of_costs.to_csv('raw_cost.csv')
prices_error.to_csv('prices_error.csv')
Theta_vector_all.to_csv('Theta_vector.csv')