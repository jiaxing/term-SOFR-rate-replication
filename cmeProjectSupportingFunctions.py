"""
Supporting functions for the CME project. 
This file should be kept in the same folder as the other scripts

Created on Fri Nov 18 11:32:06 2022
@author: riccardo
"""

import numpy as np
import pandas as pd

# =============================================================================
# # functions for processing dates
# =============================================================================

# calculate a forward date in months or years
def add_time(start, days=0, weeks=0, months=0, years=0):
    """ 
    Function for adding time to a date. 
    
    Parameters
    ----------
    start : datetime.datetime object
        The date to start from.
    days : int, optional
        The number of days to add. The default is 0.
    weeks : int, optional
        The number of weeks to add. The default is 0.
    months : int, optional
        The number of months to add. The default is 0.
    years : int, optional
        The number of years to add. The default is 0.

    Returns
    -------
    end : datetime.datetime object
        The updated date.
    """    
    import datetime
    
    updating_date = start
    updating_date += datetime.timedelta(days=days)
    updating_date += datetime.timedelta(weeks=weeks)
    
    end_year = updating_date.year + years + int((updating_date.month - 1 + months) / 12)
    end_month = (updating_date.month - 1 + months) % 12 + 1
    end_day = min(updating_date.day, pd.Timestamp(end_year, end_month, 1).daysinmonth)

    # end = datetime.datetime(end_year, end_month, updating_date.day)
    end = datetime.datetime(end_year, end_month, end_day)

    return end


# calculate the modified following date
def modfol(day, acceptable_date_range):
    """
    Function for calculating the modified following date.

    Parameters
    ----------
    day : datetime.datetime object
        The date to start from.

    acceptable_date_range : list of datetime.datetime objects
        The list of dates that are acceptable for the modified following date.

    Returns
    -------
    modfol_day : datetime.datetime object
        The modified following date.
    """
    modfol_day = np.min([date for date in acceptable_date_range if date >= day])
    # month end case 
    if modfol_day.month != day.month:
         modfol_day = np.max([date for date in acceptable_date_range if date <= day])

    return modfol_day


# =============================================================================
# # functions for processing futures
# =============================================================================

# generate all the possible futures names that might be needed
def generate_futures_codes(years=['2', '3', '4']):
    """
    Function for generating all the possible futures names that might be needed.

    Parameters
    ----------
    years : list of strings, optional
        The years to consider. The default is ['2', '3', '4'].
    
    Returns
    -------
    futures : list of strings
        The list of futures names.
    """

    futures_code = 'SER'
    futures_maturities = ['F', 'G', 'H', 'J', 'K', 'M', 'N', 'Q', 'U', 'V', 'X', 'Z']
    montly_futures = [futures_code + month + year for year in years
                      for month in futures_maturities]
    
    futures_code = 'SFR'
    futures_maturities = ['H', 'M', 'U', 'Z']
    quarterly_futures = [futures_code + month + year for year in years
                         for month in futures_maturities]
    
    futures = quarterly_futures + montly_futures

    return futures

    
# work out the start and end swap date for each future
# produce a DataFrame that contains all specificifactions all the futures
def generate_futures_specifications(years=['2', '3', '4']):
        """
        Function for generating the specifications of all the futures.

        Parameters
        ----------
        years : list of strings, optional
            The years to consider. The default is ['2', '3', '4'].
        
        Returns
        -------
        fut_specs_df : pd.DataFrame
            The DataFrame containing the specifications of all the futures.
        """
            
        import datetime
        
        futures_code_to_duration = {'SER':1, 'SFR':3}
        futures_code_to_type = {'SER':'m', 'SFR':'q'}
        futures_maturity_to_month = {'F':1, 'G':2, 'H':3, 'J':4, 'K':5, 'M':6,
                                     'N':7, 'Q':8, 'U':9, 'V':10, 'X':11, 'Z':12}
        
        futures = generate_futures_codes(years=years)
        fut_specs_df = pd.DataFrame(index=futures)
        
        for future in fut_specs_df.index:
            
            future_code = future[:3]
            year = future[4]
            month = future[3]
            year_prefix = '202'
            start_year = int(year_prefix) * 10 + int(year)
            assert datetime.datetime.now() < datetime.datetime(2028,1,1), ("WARNING: "
              "The year prefix on the futures code approaches the end of the decade")
        
            duration = futures_code_to_duration[future_code]
            start_month = futures_maturity_to_month[month]
            start = datetime.datetime(start_year, start_month, 1)
            
            end = add_time(start, months=duration)
            
            if duration == 3:
                # imm date is the 3rd wed of each month, which is weekday 2
                def from1stToIMMdate(day):
                    imm_day = 2 - day.weekday()
                    imm_day += 21 + 1 if imm_day < 0 else 14 + 1
                    return imm_day
                
                start_imm_day = from1stToIMMdate(start)        
                end_imm_day = from1stToIMMdate(end)        
          
                start = datetime.datetime(start.year, start.month, start_imm_day)
                end = datetime.datetime(end.year, end.month, end_imm_day)
            
            fut_specs_df.loc[future, 'Type'] = futures_code_to_type[future_code]
            fut_specs_df.loc[future, 'Start'] = start
            fut_specs_df.loc[future, 'End'] = end
        
        return fut_specs_df

# identify the futures relevant for a specific day
def identify_relevant_futures(day, fut_specs_df):
    """
    Function for identifying the futures relevant following CME Group's approach
    Modify it as needed if your strategy includes different futures

    Parameters
    ----------
    day : datetime.datetime object
        The date to start from.
    fut_specs_df : pd.DataFrame
        The DataFrame containing the specifications of all the futures.

    Returns
    -------
    relevant_futures : list of strings
        The list of futures relevant for the specific day.
    """
    import datetime
    
    sfrs = fut_specs_df[fut_specs_df['Type'] == 'q']
    sfrs = sfrs[sfrs['End'] >= day + datetime.timedelta(3)]
    sfrs = sfrs.iloc[:5]
    assert len(sfrs) == 5, "In identify relevant futures, the number of SFR is not 5"
    
    sers = fut_specs_df[fut_specs_df['Type'] == 'm']
    sers = sers[sers['End'] >= day + datetime.timedelta(3)]
    sers = sers.iloc[:13]    
    assert len(sers) == 13, "In identify relevant futures, the number of SER is not 13"
    
    relevant_futures = list(sfrs.index) + list(sers.index)
    
    return relevant_futures
    
