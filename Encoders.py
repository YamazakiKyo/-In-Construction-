from tqdm import tqdm
from time import sleep
import pandas as pd
from sklearn.preprocessing import LabelEncoder, normalize

# features = ['ALIGNMENT_CD', 'CRASH_YEAR', 'CR_MONTH', 'DAY_OF_WK', 'DR_SEX',
#             'HIT_AND_RUN', 'LIGHTING_CD', 'LOC_TYPE_CD', 'PEDESTRIAN',
#             'PRI_CONTRIB_FAC_CD', 'SEC_CONTRIB_FAC_CD','ROAD_COND_CD', 'ROAD_TYPE_CD',
#             'SURF_COND_CD', 'SURF_TYPE_CD', 'WEATHER_CD', 'city_level', 'age_group',
#             'crash_time', 'vehicle_type', 'From_Louisiana', 'vehicle_year',
#             'num_occupant', 'num_vehicle', 'aggressive_driving',
#             'drive_condition', 'drive_distract', 'manner_of_collision']
features = ['CR_MONTH', 'ALIGNMENT_CD', 'PRI_CONTRIB_FAC_CD',
       'SEC_CONTRIB_FAC_CD', 'DAY_OF_WK', 'HIT_AND_RUN', 'LIGHTING_CD',
       'LOC_TYPE_CD', 'NUM_VEH', 'ROAD_COND_CD', 'ROAD_REL_CD', 'ROAD_TYPE_CD',
       'SURF_COND_CD', 'SURF_TYPE_CD', 'WEATHER_CD',
       'PEDESTRIAN', 'DR_SEX', 'From_Louisiana', 'NUM_OCC',
        'city_level', 'highway_type', 'manner_of_collision',
       'aggressive_driving', 'age_group', 'crash_time', 'vehicle_type',
       'drive_distract', 'drive_condition', 'vehicle_year']


year = 2014

df_train = pd.read_csv('./data_cleaned/' + str(year) + '_cleaned.csv')
X = df_train[features]
y = df_train['SEVERITY_CD']
y = y.fillna('NaN')
y = LabelEncoder().fit_transform(y)
int_list = ['CR_MONTH', 'HIT_AND_RUN', 'NUM_VEH','PEDESTRIAN', 'age_group',
            'crash_time', 'NUM_OCC', 'vehicle_year']
for col in tqdm(int_list):
    X[str(col)] = X[str(col)].dropna().apply(lambda x: int(x))
    X[str(col)]=X[str(col)].fillna(-1)
    # X[str(col)] = LabelEncoder().fit_transform(X[str(col)])
    X[str(col)] = X[str(col)].astype(int)
    sleep(0.1)

obj_list = []
for items in tqdm(X.columns):
    if items not in int_list:
        obj_list.append(items)

for items in tqdm(obj_list):
    X[str(items)]=X[str(items)].fillna('NaN')
    X[str(items)] = LabelEncoder().fit_transform(X[str(items)])
    sleep(0.1)

X.to_csv('./data_encoded/' + str(year) + '_data.csv', index=False)
y_train_df = pd.DataFrame(y)
y_train_df.to_csv('./data_encoded/'+str(year)+'_label.csv', index=False)
