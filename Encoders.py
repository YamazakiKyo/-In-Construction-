from tqdm import tqdm
from time import sleep
import pandas as pd
from sklearn.preprocessing import LabelEncoder

features = ['ALIGNMENT_CD', 'CRASH_YEAR', 'CR_MONTH', 'DAY_OF_WK', 'DR_SEX',
            'HIT_AND_RUN', 'LIGHTING_CD', 'LOC_TYPE_CD', 'PEDESTRIAN',
            'PRI_CONTRIB_FAC_CD', 'ROAD_COND_CD', 'ROAD_TYPE_CD',
            'SURF_COND_CD', 'SURF_TYPE_CD', 'WEATHER_CD', 'city_level', 'age_group',
            'crash_time', 'vehicle_type', 'From_Louisiana', 'vehicle_year',
            'num_occupant', 'num_vehicle', 'aggressive_driving',
            'drive_condition', 'drive_distract', 'manner_of_collision']

def encoder(year):
    df_train = pd.read_csv('./' + str(year) + '_cleaned.csv')
    X_train = df_train[features]
    y_train = df_train['SEVERITY_CD']
    y_train = y_train.fillna('NaN')
    y_train = LabelEncoder().fit_transform(y_train)
    int_list = ['CRASH_YEAR', 'CR_MONTH', 'HIT_AND_RUN', 'PEDESTRIAN', 'age_group', 'crash_time', 'aggressive_driving']
    for col in tqdm(int_list):
        X_train[str(col)] = X_train[str(col)].dropna().apply(lambda x: int(x))
        X_train[str(col)]=X_train[str(col)].fillna(-1)
        X_train[str(col)] = LabelEncoder().fit_transform(X_train[str(col)])
        sleep(0.1)

    for items in tqdm(X_train.columns):
        if X_train.dtypes[str(items)] == 'object':
            X_train[str(items)]=X_train[str(items)].fillna('NaN')
            X_train[str(items)] = LabelEncoder().fit_transform(X_train[str(items)])
        sleep(0.1)

    X_train.to_csv('./data_encoded/'+str(year)+'_data.csv', index=False)
    y_train_df = pd.DataFrame(y_train)
    y_train_df.to_csv('./data_encoded/'+str(year)+'_label.csv', index=False)

for yr in range(2006, 2007):
    encoder(yr)