"""
Created on Mon Feb 11 2019

Contains functions:
    - calc_soundspeed
    - add_soundspeed
    - prepare_tag_data
    - clean_toa_data
    - fill_gaps
    - create_final_toa
    - create_plots

@authors: Thomas Vanwyck & Jenna Vergeynst
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob

def calc_soundspeed(T, S=0, z=1):
    """
    Equation of Mackenzie: http://asa.scitation.org/doi/10.1121/1.386919
    parameters
    ----------
    T = temperatuur in °C
    S = salinity in parts per thousand
    z = depth in meters

    returns
    ------
    soundspeed in m/s
    """

    a1 = 1448.96
    a2 = 4.591
    a3 = -5.304e-2
    a4 = 2.374e-4
    a5 = 1.34
    a6 = 1.63e-2
    a7 = 1.675e-7
    a8 = -1.025e-2
    a9 = -7.139e-13

    SS = a1 + a2*T + a3*T**2 + a4*T**3 + a5*(S-35) + a6*z + a7*z**2 + a8*T*(S-35) + a9*T*z**3
    return SS

def add_soundspeed(tag_data, time_col, temp, temp_time_col, temp_temp_col):
    """
    Function to add max_time to the dataframe. The maximum travel time of a sound signal
    of one observation, between to successive receivers, is the travel time between these receivers.

    Inputs
    ------
    tag_data : DataFrame
        contains all receiver detections for one fish tag
    time_col : string
        name of SyncTime column
    temp : DataFrame
        contains temperature information (each 10 minutes)
    temp_time_col : string
        name of time column in temp DataFrame
    temp_temp_col : string
        name of temperature column in temp DataFrame


    Output
    ------
    tag_data : DataFrame
        has column 'soundspeed' added.
    """

    tag_data = pd.merge_asof(tag_data, temp[[temp_time_col, temp_temp_col]], left_on=time_col, right_on=temp_time_col, direction='nearest')
    tag_data['soundspeed'] = calc_soundspeed(T=tag_data[temp_temp_col])
    return tag_data


def prepare_tag_data(input_data, time_col, rec_col, max_time, pas_tol):
    """
    Function to rearrange detections to dataframe with 1 observation per row.
    Observations are split up in different passings if time between observations is too long

    Inputs
    ------
    tag_data : DataFrame
        Contains all receiver detections for one fish tag
    time_col : string
        name of SyncTime column
    rec_col : string
        name of receiver column
    max_time : Float
        Maximum time (in seconds) that a sound signal would take between furthest receivers
    pas_tol : int
        Max minutes between observations before a new passing is started and the track is split up

    Returns
    -------
    toa_data : DataFrame
         TOA matrix usable by YAPS and TDOA positioning
         Columns are the receivers, rows are the observations, split up into passings
         Also contains a soundspeed and SyncTime column
    """
    tag_data = input_data.copy()
    tag_data['time_diff'] = tag_data[time_col].diff()/pd.Timedelta(seconds=1)
    tag_data['sec_since_start'] = tag_data['time_diff'].cumsum().fillna(0)
    # make gaps when time diff > max time between furthest receivers
    gaps = tag_data[time_col].diff() > pd.Timedelta(seconds=max_time)
    # cumsum of falses and trues creates groups
    tag_data['groups_obs'] = gaps.cumsum()
    # idem for tracks
    gaps2 = tag_data[time_col].diff() > pd.Timedelta(minutes = pas_tol)
    tag_data['groups_pas'] = gaps2.cumsum()
    # save soundspeed for YAPS model and synced time for splitting in tracks
    soundspeed = tag_data.set_index(['groups_obs'])['soundspeed'].groupby('groups_obs').mean()
    SyncTime = tag_data.set_index(['groups_obs'])[time_col].groupby('groups_obs').first()
    # reshape the resulting dataframe
    toa_data = tag_data.set_index(['groups_pas','groups_obs',rec_col])['sec_since_start'].unstack()
    # put back soundspeed and synced time
    toa_data.columns = toa_data.columns.astype(str)
    toa_data['soundspeed'] = soundspeed.values
    toa_data[time_col]= SyncTime.values

    return toa_data



def clean_toa_data(toa_data,min_delay, rec_cols):
    '''
    Cleans up the TOA matrix, created with prepare_tag_data.
    When observations follow eachother faster than the minimum known time delay,
    the observation with the least receivers is removed.
    Afterwards, the remaining observations that are still too close,
    are dropped if they have only one receiver in the row.
    Reasoning: these observations are too far apart to belong to the same row
    (i.e. further than max travel time between receivers), so are probably originating from a multipath.

    Parameters
    ----------
    toa_data : DataFrame
        The TOA matrix. Output of the prepare_tag_data.
    min_delay : int or float
        The minimum time delay for the used transmitter, defined by the manufacturer
    rec_cols : list
        List of receiver names in the columns

    Returns
    -------
    toa_data_cleaned : DataFrame
        Cleaned dataframe, with unreliable observations removed
    '''
    #remove soundspeed and SyncTime column for analysis
    toa_data_cleaned = toa_data.copy()
    toa_data_cleaned['t_diff'] = toa_data_cleaned.loc[:,rec_cols].mean(axis = 1).diff().values
    toa_data_cleaned['receiver_amount'] = toa_data_cleaned.loc[:,rec_cols].count(axis = 1).values
    #check if second point of impossible interval is the wrong one
    #define second point of impossible interval
    #see if last point was picked up by more receivers
    toa_data_cleaned['second_wrong'] = ((toa_data_cleaned['receiver_amount'].shift(1)>toa_data_cleaned.receiver_amount)
                                        & (toa_data_cleaned['t_diff']<min_delay))
    #check if first point of impossible interval is the wrong one
    #define first point of impossible interval
    #see if next point was picked up by more receivers
    toa_data_cleaned['first_wrong'] = ((toa_data_cleaned['receiver_amount'].shift(-1)>toa_data_cleaned.receiver_amount)
                                        & (toa_data_cleaned['t_diff'].shift(-1)<min_delay))
    toa_data_cleaned['true_error'] = toa_data_cleaned.first_wrong| toa_data_cleaned.second_wrong
    # put back soundspeed and SyncTime column
    toa_data_cleaned = toa_data[~toa_data_cleaned.true_error].copy()

    # Second loop: delete remaining rows of impossible intervals that only have one receiver
    toa_data_cleaned['t_diff'] = toa_data_cleaned.loc[:,rec_cols].mean(axis = 1).diff().values
    toa_data_cleaned['receiver_amount'] = toa_data_cleaned.loc[:,rec_cols].count(axis = 1).values
    trash_indices = toa_data_cleaned[(toa_data_cleaned['t_diff']<min_delay)&(toa_data_cleaned['receiver_amount']==1)].index
    toa_data_cleaned.drop(index=trash_indices, columns=['t_diff', 'receiver_amount'], inplace=True)
    toa_data_cleaned.reset_index(drop=True, inplace=True)

    return toa_data_cleaned



def fill_gaps(toa_part, rec_cols, time_col, mean_burst):
    """
    Function to fill the gaps of missing pings in the dataframe, based on R code of Henrik Baktoft.
    Based on the average ping interval, a "virtual ping"-timeseries is created with stable burst interval on every average ping.
    Then the available pings are related to their closest virtual ping, and all virtual pings between get NaN-rows.

    toa_part : DataFrame
        part of toa_data belonging to one pas (or track)
    rec_cols : list
        List of receiver names in the columns
    time_col : string
        name of time column
    mean_burst: string
        mean burst interval of the fish transmitter e.g. '1.2s'

    """

    # make df with only indexes, containing the times if ping would be heard at every burst
    mean_bi_times = pd.date_range(start=toa_part[time_col].min(), end=toa_part[time_col].max()+pd.Timedelta('1s'), freq=pd.Timedelta(mean_burst))
    mean_bi_df = pd.DataFrame(index=mean_bi_times)
    # insert virtual ping times in the gaps
    toa_part = toa_part.set_index(time_col, drop=False)
    toa_filled = toa_part.reindex(mean_bi_df.index, method='nearest')
    # now all virtual ping times are connected to the real 'nearest' ping,
    # but all virtual pings except the ones with the closest 'nearest ping' should be NaN rows.
    # therefore split the df in a df containing all the gaps (so duplicate rows), and a df without duplicates
    toa_filled_dup = toa_filled[toa_filled.duplicated(keep=False)].copy() # keep = False puts all the duplicate rows on True (not only first or last)
    toa_filled_nondup = toa_filled[~toa_filled.duplicated(keep=False)].copy()
    # calculate the offset of real versus virtual ping (sbi)
    toa_filled_dup['offset_sbi'] = (toa_filled_dup[time_col]-toa_filled_dup.index).dt.total_seconds().abs()
    # group in groups of duplicates
    dup_groups = toa_filled_dup.groupby(by='groups_obs')
    # per group, find the index of the row that contains the real ping closest to the virtual ping
    # this is the longest part, about 1min for a 1GB df
    tokeep_indices = dup_groups.apply(lambda x: x.offset_sbi.idxmin())
    # all rows that are not in tokeep_indices have to be NaN in receiver columns and time_col
    toa_filled_dup.loc[toa_filled_dup.index.isin(tokeep_indices)==False,rec_cols+[time_col]] = np.nan
    # concatenate dups and nondups again and sort
    result = pd.concat([toa_filled_dup, toa_filled_nondup], axis=0, sort=True).sort_index()
    result = result.reset_index(drop=False)
    # fill in nans of synced time with index time
    result.loc[result[time_col].isna(), time_col] = result.loc[result[time_col].isna(), 'index']
    result.drop(columns=['offset_sbi', 'index'], inplace=True)

    return result



def create_final_toa(tag_data,  max_time, min_burst, max_burst, time_col,
                     rec_col, rec_list, temp, temp_time_col, temp_temp_col,
                     pas_tol=5, min_track_length=120):
    """
    Function to create final toa dataframe.

    tag_data : DataFrame
        Contains all detections of one fish on all receivers
    max_time : Float
        Maximum time (in seconds) that a sound signal would take between furthest receivers
    min_burst : float
        min burst interval of fish transmitter
    max_burst : float
        max burst interval of fish transmitter
    time_col : string
        name of time column
    rec_col : string
        name of receiver column
    rec_list : List
        List with all receiver names (e.g. uniques from receivers in tag_data)
    temp : DataFrame
        contains temperature information (each 10 minutes)
    temp_time_col : string
        name of time column in temp DataFrame
    temp_temp_col : string
        name of temperature column in temp DataFrame
    pas_tol : int
        Max minutes between observations before a new passing is started and the track is split up
    min_track_lenght : int
        min lenght of a track

    Returns
    -------
    rec_cols : list
        List of receiver names in the columns, for the receivers that observed this fish (needed for plotting)
    toa_data : DataFrame
        tag_data rearranged for yaps without any cleaning
    cleaned_toa_data : DataFrame
        tag_data with rows that succeed each other faster than min delay removed
    filled_toa : dictionnary
        contains for each groups_pas nb a DataFrame of which gaps are filled
    final_toa : dictionnary
        contains for each groups_pas nb the DataFrame after the second round of gap filling


    """

    mean_burst = str(np.mean((min_burst,max_burst)).round(1))+'s'

    tag_data = tag_data.sort_values(by=time_col).reset_index(drop=True)

    tag_data = add_soundspeed(tag_data=tag_data, time_col=time_col, temp=temp,
                              temp_time_col=temp_time_col, temp_temp_col=temp_temp_col)

    # pas_tol = 5: cut track when a fish is not heard during 5 minutes. Don't make this too small
    toa_data = prepare_tag_data(tag_data, time_col=time_col, rec_col=rec_col, max_time=max_time, pas_tol=pas_tol).reset_index()
    rec_cols = [x for x in toa_data.columns if x in rec_list]

    # clean observations that follow eachother faster than the minimum known time delay (allow 10% lower than min)
    cleaned_toa_data = clean_toa_data(toa_data,min_delay=0.9*min_burst, rec_cols=rec_cols)

    # fill the gaps of pings that were not observed
    toa_data_groups = cleaned_toa_data.groupby(by='groups_pas')
    filled_toa = {}
    for group_key in toa_data_groups.groups.keys():
        toa_part = toa_data_groups.get_group(group_key).copy()
        # only consider a track if it is more than 2 minutes long (120 seconds). Too short tracks will bug YAPS.
        if len(toa_part)>min_track_length:
            result = fill_gaps(toa_part, rec_cols, time_col=time_col, mean_burst=mean_burst)
            filled_toa[group_key] = result

    # Some burst intervals are still double of max interval,
    # probably due to use of virtual pings and incorrect alignment with real pings.
    # Remove the pings creating these intervals
    for key in filled_toa.keys():
        # allow interval to be 10% more than max_burst
        errors = filled_toa[key][filled_toa[key].loc[:,rec_cols].mean(axis=1).diff()>1.1*max_burst]
        filled_toa[key].drop(index=errors.index, inplace=True)

    # This creates again gaps in the data, so repeat the gaps filling
    final_toa = {}
    for group_key in filled_toa.keys():
        toa_part = filled_toa[group_key].copy()
        # only consider a track if it is more than min_track_length long. Too short tracks will bug YAPS.
        if len(toa_part)>min_track_length:
            result = fill_gaps(toa_part, rec_cols, time_col=time_col, mean_burst=mean_burst)
            final_toa[group_key] = result

    return rec_cols, toa_data, cleaned_toa_data, filled_toa, final_toa



def create_plots(ID, min_burst, max_burst, rec_cols, toa_data, cleaned_toa_data, filled_toa, final_toa, write_path):

    """
    Make plots for checking quality of TOA matrix

    ID : string
        ID of fish
    rec_cols : list
        List of receiver names in the columns, for the receivers that observed this fish (needed for plotting)
    toa_data : DataFrame
        tag_data rearranged for yaps without any cleaning
    cleaned_toa_data : DataFrame
        tag_data with rows that succeed each other faster than min delay removed
    filled_toa : dictionnary
        contains for each groups_pas nb a DataFrame of which gaps are filled
    final_toa : dictionnary
        contains for each groups_pas nb the DataFrame after the second round of gap filling
    write_path : string
        path where to save figures


    """
    # don't show plots in notebook
    plt.ioff()

    fig,ax = plt.subplots()
    toa_data.loc[:,rec_cols].mean(axis=1).diff().plot(marker='.', lw=0, ax=ax, alpha=0.3)
    ax.set_ylim(-5,5*max_burst)
    ax.plot(ax.get_xbound(),(min_burst, min_burst), c='red')
    ax.plot(ax.get_xbound(),(max_burst, max_burst), c='red')
    ax.set_title('Burst intervals fish '+str(ID)+' without cleaning')
    # should remove intervals below min interval
    fig.savefig(write_path+'1_uncleaned_'+str(ID)+'.png')

    fig,ax = plt.subplots()
    cleaned_toa_data.loc[:,rec_cols].mean(axis=1).diff().plot(marker='.', lw=0, ax=ax, alpha=0.3)
    ax.set_ylim(-5,5*max_burst)
    ax.plot(ax.get_xbound(),(min_burst, min_burst), c='red')
    ax.plot(ax.get_xbound(),(max_burst, max_burst), c='red')
    ax.set_title('Burst intervals fish '+str(ID)+' after first clean-up of too-close rows')
    # should remove intervals below min interval
    fig.savefig(write_path+'2_cleaned_'+str(ID)+'.png')

    fig,ax = plt.subplots()
    for key in filled_toa.keys():
        filled_toa[key].loc[:,rec_cols].mean(axis=1).diff().plot(marker='.', lw=0, ax=ax, alpha=0.3)
    ax.set_ylim(0,5*max_burst)
    ax.plot(ax.get_xbound(),(min_burst, min_burst), c='red')
    ax.plot(ax.get_xbound(),(max_burst, max_burst), c='red')
    ax.set_title('Burst intervals fish '+str(ID)+ ' after filling of gaps')
    # should remove major part of intervals above max interval
    fig.savefig(write_path+'3_gap_filled_'+str(ID)+'.png')

    fig,ax = plt.subplots()
    for key in final_toa.keys():
        final_toa[key].loc[:,rec_cols].mean(axis=1).diff().plot(marker='.', lw=0, ax=ax, alpha=0.3)
    #ax.set_ylim(0,5)
    ax.plot(ax.get_xbound(),(min_burst, min_burst), c='red')
    ax.plot(ax.get_xbound(),(max_burst, max_burst), c='red')
    ax.set_title('Burst intervals fish '+str(ID)+ ' after second round of gap filling')
    # Now all pings should be between min and max
    fig.savefig(write_path+'4_final_'+str(ID)+'.png')

    return
