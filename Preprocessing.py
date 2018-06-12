from tqdm import tqdm
from time import sleep # A tool of progress bar for time control
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize

def del_col(df, col):
    '''delete one column in dataFrame with inplacement'''
    df.drop(str(col), axis = 1, inplace = True)

def TenYrsData(yr):
    '''read from the 10 years' crash database'''
    # df2006 = pd.read_csv('./data/crash2006.csv')
    # df2007 = pd.read_csv('./data/crash2007.csv')
    # df2008 = pd.read_csv('./data/crash2008.csv')
    # df2009 = pd.read_csv('./data/crash2009.csv')
    # df2010 = pd.read_csv('./data/crash2010.csv')
    # df2011 = pd.read_csv('./data/crash2011.csv')
    # df2012 = pd.read_csv('./data/crash2012.csv')
    # df2013 = pd.read_csv('./data/crash2013.csv')
    # df2014 = pd.read_csv('D:\PyProject\Distraction_Affected_Crashes\data\crash2014.csv')
    # df2015 = pd.read_csv('./data/crash2015.csv')
    # frames = [df2006, df2007, df2008, df2009, df2010, df2011, df2012, df2013, df2014, df2015]
    # df = pd.concat(frames)
    df = pd.read_csv('./data/crash' + str(yr) + '.csv')
    return df

def PopCollect():
    '''Collect the Louisiana population census data from wikipedia'''
    url = 'https://simple.wikipedia.org/wiki/List_of_cities,_towns,_and_villages_in_Louisiana'
    df_city = pd.read_html(url, header=0)
    df_city = df_city[0].dropna(axis=0, thresh=4)  # Transfer df_city from 'array list' to 'data frame'
    return df_city


def CityLevel(row):
    '''Classify the cities by different population'''
    if row['Population (2010)[1]'] > 100000:
        return 'A'
    if (row['Population (2010)[1]'] <= 100000) and (row['Population (2010)[1]'] > 30000):
        return 'B'
    if (row['Population (2010)[1]'] <= 30000) and (row['Population (2010)[1]'] > 10000):
        return 'C'
    if (row['Population (2010)[1]'] <= 10000) and (row['Population (2010)[1]'] > 5000):
        return 'D'
    if row['Population (2010)[1]'] <= 5000:
        return 'E'
    else:
        return 'F'


def AgeGroup(row):
    '''Classify the drivers by different age group'''
    # if (row['DR_AGE'] >= 0) and (row['DR_AGE'] <= 25):
    #     return 'A'
    # if (row['DR_AGE'] > 25) and (row['DR_AGE'] <= 35):
    #     return 'B'
    # if (row['DR_AGE'] > 35) and (row['DR_AGE'] <= 45):
    #     return 'C'
    # if (row['DR_AGE'] > 45) and (row['DR_AGE'] <= 55):
    #     return 'D'
    # if (row['DR_AGE'] > 55) and (row['DR_AGE'] <= 65):
    #     return 'E'
    # if (row['DR_AGE'] > 65) and (row['DR_AGE'] <= 199):
    #     return 'F'
    # return np.NaN
    if (row['DR_AGE'] > 100) or (row['DR_AGE'] < 0):
        return np.NaN
    else:
        return row['DR_AGE']


def CrashTime(row):
    '''Classify the crashes occurred time by group'''
    # if (row['CR_HOUR'] >= 4) and (row['CR_HOUR'] <= 7):
    #     return 'A'
    # if (row['CR_HOUR'] > 7) and (row['CR_HOUR'] <= 11):
    #     return 'B'
    # if (row['CR_HOUR'] > 11) and (row['CR_HOUR'] <= 15):
    #     return 'C'
    # if (row['CR_HOUR'] > 15) and (row['CR_HOUR'] <= 19):
    #     return 'D'
    # if (row['CR_HOUR'] > 19) and (row['CR_HOUR'] <= 23):
    #     return 'E'
    # if (row['CR_HOUR'] > 23) or (row['CR_HOUR'] <= 3):
    #     return 'F'
    # return np.NaN
    if (row['CR_HOUR'] > 23) or (row['CR_HOUR'] < 0):
        return np.NaN
    else:
        return row['CR_HOUR']


def VehicleType(row):
    '''Classify the vehicle types by group'''
    abnVT = ['J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'T', 'V']
    for items in abnVT:
        if row['VEH_TYPE_CD'] == str(items):
            return np.NaN
        else:
            return row['VEH_TYPE_CD']

def DRState(row):
    ''' Clearn the DR_STATE column '''
    usStateAbbv = ['AK', 'AL','AR','AZ', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
                   'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA',
                   'MD', 'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE',
                   'NH', 'NJ', 'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI',
                   'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']
    try:
        row['DR_STATE'] = row['DR_STATE'].upper()
    except:
        print('No str found!')
    for state in usStateAbbv:
        if row['DR_STATE'] == str(state):
            return row['DR_STATE']
        else:
            return np.NaN



def VehicleYear(row):
    '''Classify the vehicle year by group'''
    # if (row['VEH_YEAR'] >= 1600) and (row['VEH_YEAR'] <= 1985):
    #     return 'A'
    # if (row['VEH_YEAR'] > 1985) and (row['VEH_YEAR'] <= 1990):
    #     return 'B'
    # if (row['VEH_YEAR'] > 1990) and (row['VEH_YEAR'] <= 1995):
    #     return 'C'
    # if (row['VEH_YEAR'] > 1995) and (row['VEH_YEAR'] <= 2000):
    #     return 'D'
    # if (row['VEH_YEAR'] > 2000) and (row['VEH_YEAR'] <= 2005):
    #     return 'E'
    # if (row['VEH_YEAR'] > 2005) and (row['VEH_YEAR'] <= 2010):
    #     return 'F'
    # return 'G'
    if (row['VEH_YEAR'] > 2018) or (row['VEH_YEAR'] <1950):
        return np.NaN
    else:
        return row['VEH_YEAR']


def MergePop(df, df_city):
    '''Classify the cities' level, and merge with 10 years data'''
    df_city['city_level'] = df_city.apply(lambda row: CityLevel(row), axis=1)  # Apply func 'CityLevel' to each row
    df['CITY'] = df['CITY'].str.lower()
    df_city['Name'] = df_city['Name'].str.lower()
    # df_city['Name'] = df_city['Name'].map(lambda x:x.rstrip('[e]' ))
    # This is to remove the suffix, not work, because if will substitute 'lafayette[e]' with 'lafayett'
    df_city['Name'] = df_city['Name'].str.split('[').str[
        0]  # If the suffix starts with '[', split the column and only pick the preffix
    df = df.merge(df_city[['Name', 'city_level']], left_on='CITY', right_on='Name', how='left')
    del_col(df, 'Name')
    return df


def noiseRemover(df):
    '''Remove mistaken coded data, prepare for passing to the algorithm'''
    for i in tqdm(['Y', 'Z']):
        df['ROAD_TYPE_CD'] = df['ROAD_TYPE_CD'].replace(str(i), np.NaN)
        df['ALIGNMENT_CD'] = df['ALIGNMENT_CD'].replace(str(i), np.NaN)
        df['ROAD_COND_CD'] = df['ROAD_COND_CD'].replace(str(i), np.NaN)
        df['LIGHTING_CD'] = df['LIGHTING_CD'].replace(str(i), np.NaN)
        df['SURF_COND_CD'] = df['SURF_COND_CD'].replace(str(i), np.NaN)
        df['WEATHER_CD'] = df['WEATHER_CD'].replace(str(i), np.NaN)
        sleep(0.1)
    for m in tqdm(['F', 'I', 'J', 'K', 'L', 'Y', 'Z', 'G', 'H', 'M']):
        df['SURF_TYPE_CD'] = df['SURF_TYPE_CD'].replace(str(m), np.NaN)
        sleep(0.1)
    # for i in tqdm(['B', 'D', 'E', 'F']):
    #     df.loc[df['LIGHTING_CD'] == str(i), 'LIGHTING_CD'] = 'B'
    df['LOC_TYPE_CD'] = df['LOC_TYPE_CD'].replace('Z', np.NaN)
    df['DR_SEX'] = df['DR_SEX'].replace('U', np.NaN)
    df['DR_SEX'] = df['DR_SEX'].replace('W', 'F')
    df['DR_SEX'] = df['DR_SEX'].replace('B', 'M')
    df['CRASH_YEAR'] = df['CRASH_YEAR'].replace(1901, np.NaN)
    df['CRASH_YEAR'] = df['CRASH_YEAR'].replace(2005, 2006)
    df['CRASH_YEAR'] = df['CRASH_YEAR'].replace(2004, 2006)
    df['CRASH_YEAR'] = df['CRASH_YEAR'].replace(2000, np.NaN)
    df['CRASH_YEAR'] = df['CRASH_YEAR'].replace(2002, np.NaN)
    df['CRASH_YEAR'] = df['CRASH_YEAR'].replace(1936, np.NaN)
    # for i in tqdm(['A', 'B', 'D', 'H', 'I']):
    #     df['PRI_CONTRIB_FAC_CD'] = df['PRI_CONTRIB_FAC_CD'].replace(str(i), np.NaN)
    #     df['SEC_CONTRIB_FAC_CD'] = df['SEC_CONTRIB_FAC_CD'].replace(str(i), np.NaN)
    #     sleep(0.1)
    # for j in tqdm(['C', 'E', 'F', 'G', 'J', 'K', 'L', 'M']):
    #     df.loc[(df['PRI_CONTRIB_FAC_CD'].isnull == True) & (
    #                 df['SEC_CONTRIB_FAC_CD'] == str(j)), 'PRI_CONTRIB_FAC_CD'] = str(j)
    #     sleep(0.1)

    '''Highway type classification'''
    for item in tqdm(['A', 'B', 'C', 'D', 'E', 'G']):
        df.loc[df['HWY_TYPE_CD'].str.contains(str(item), case=False) == True, 'highway_type'] = str(item)
        sleep(0.1)

    '''Manner of collision (crash type)'''
    for item in tqdm(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']):
        df.loc[df['MAN_COLL_CD'].str.contains(str(item), case=False) == True, 'manner_of_collision'] = str(item)
        sleep(0.1)
    df.loc[(df['manner_of_collision'] == 'E') | (df['manner_of_collision'] == 'F') | (
                df['manner_of_collision'] == 'G'), 'manner_of_collision'] = 'E'
    df.loc[(df['manner_of_collision'] == 'H') | (df['manner_of_collision'] == 'I'), 'manner_of_collision'] = 'F'
    df.loc[(df['manner_of_collision'] == 'J') | (df['manner_of_collision'] == 'K'), 'manner_of_collision'] = 'G'

    '''Drivers from the Louisiana state or not'''
    df['DR_STATE'] = df['DR_STATE'].str.upper()
    df.loc[(df['DR_STATE'] == 'LA') | (df['DR_STATE'] == 'LOUISIANA'), 'From_Louisiana'] = 'Yes'
    df.loc[(df['DR_STATE'] != 'LA') & (df['DR_STATE'] != 'LOUISIANA'), 'From_Louisiana'] = 'No'
    # df['DR_STATE'] = df.apply(lambda row: DRState(row), axis=1)

    '''Number of people in the vehicle when the crashes occurred'''
    # df.loc[df['NUM_OCC'].astype(float) == 1, 'num_occupant'] = 'One'
    # df.loc[df['NUM_OCC'].astype(float) == 2, 'num_occupant'] = 'Two'
    # df.loc[df['NUM_OCC'].astype(float) == 3, 'num_occupant'] = 'Three'
    # df.loc[df['NUM_OCC'].astype(float).isin(range(1, 4)) == False, 'num_occupant'] = 'More_than_Three'

    '''Number of vehicles when the crshes occurred'''
    # df.loc[df['NUM_VEH'] == 1, 'num_vehicle'] = 'One'
    # df.loc[df['NUM_VEH'] == 2, 'num_vehicle'] = 'Two'
    # df.loc[df['NUM_VEH'] >= 3, 'num_vehicle'] = 'More_than_two'

    '''Aggressive driving behabior exists when the crashes occurred'''
    df.loc[df['agressive'] >= 1, 'aggressive_driving'] = 'Yes'
    df.loc[df['agressive'] == 0, 'aggressive_driving'] = 'No'

    '''Age group classification'''
    df['age_group'] = df.apply(lambda row: AgeGroup(row), axis=1)

    '''Crash time classification'''
    df['crash_time'] = df.apply(lambda row: CrashTime(row), axis=1)

    '''Vehicle type classification'''
    df['VEH_TYPE_CD'] = df['VEH_TYPE_CD'].str.upper()
    df['vehicle_type'] = df.apply(lambda row: VehicleType(row), axis=1)

    '''driver_distraction'''
    list_distract = ['A', 'B', 'C', 'D']
    list_condition = ['B', 'C']
    for i in tqdm(list_distract):
        df.loc[df['DR_DISTRACT_CD'] == str(i), 'drive_distract'] = str(i)
        sleep(0.1)
    df.loc[(df['DR_DISTRACT_CD'].isin(list_distract) == False) & (df['DR_COND_CD'].isin(list_condition)), 'drive_distract'] = 'E'

    '''Drivers' condition classification'''
    df.loc[df['DR_COND_CD'] == 'D', 'drive_condition'] = 'ill'
    df.loc[df['DR_COND_CD'] == 'K', 'drive_condition'] = 'ill'
    df.loc[df['DR_COND_CD'] == 'E', 'drive_condition'] = 'fatigue'
    df.loc[df['DR_COND_CD'] == 'F', 'drive_condition'] = 'fatigue'
    df.loc[df['DR_COND_CD'] == 'G', 'drive_condition'] = 'd&d'
    df.loc[df['DR_COND_CD'] == 'H', 'drive_condition'] = 'd&d'
    df.loc[df['DR_COND_CD'] == 'I', 'drive_condition'] = 'd&d'
    df.loc[df['DR_COND_CD'] == 'J', 'drive_condition'] = 'd&d'
    list_distract_num = ['G', 'H', 'I', 'J']
    df.loc[(df['DR_COND_CD'].isin(list_distract_num) == False) & (
                df['DRUGS'].astype(float) == 1), 'drive_condition'] = 'd&d'
    df.loc[(df['DR_COND_CD'].isin(list_distract_num) == False) & (
                df['ALCOHOL'].astype(float) == 1), 'drive_condition'] = 'd&d'

    '''Vehicle years classification'''
    df['VEH_YEAR'] = pd.to_numeric(df['VEH_YEAR'], errors='coerce')
    df['vehicle_year'] = df.apply(lambda row: VehicleYear(row), axis=1)

    '''Delete useless columns'''
    Useless_list = ['NUM_DRI_IK', 'NUM_DRI_INJ', 'NUM_DRI_KIL', 'NUM_OCC_IK', 'NUM_OCC_INJ',
                    'NUM_OCC_KIL', 'NUM_PED_IK', 'NUM_PED_INJ', 'NUM_PED_KIL', 'NUM_TOT_IK', 'violation',
                    'MAN_COLL_CD', 'agressive', 'CRASH_NUM',
                    'DR_AGE', 'CR_HOUR', 'VEH_TYPE_CD', 'VEH_YEAR', 'HWY_TYPE_CD', 'DR_DISTRACT_CD',
                    'DR_COND_CD', 'DRUGS', 'ALCOHOL', 'NUM_TOT_INJ', 'NUM_TOT_KIL']
    for ul in Useless_list:
        del_col(df, str(ul))
    return df

for yr in range(2014, 2015):
    df_10yrs = TenYrsData(yr)
    df_cityLevel = PopCollect()
    df_levelMerged = MergePop(df_10yrs, df_cityLevel)
    df_cleaned = noiseRemover(df_levelMerged)

    df_cleaned.to_csv('./data_cleaned/' + str(yr) + '_cleaned.csv', index = False)

    for xxx in df_cleaned.columns:
        print(df_cleaned[str(xxx)].value_counts())
