from tqdm import tqdm
from time import sleep
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import ictaiDefs.DataInput as iD

year = 2014


features = ['CR_MONTH', 'ALIGNMENT_CD', 'PRI_CONTRIB_FAC_CD',
       'SEC_CONTRIB_FAC_CD', 'DAY_OF_WK', 'HIT_AND_RUN', 'LIGHTING_CD',
       'LOC_TYPE_CD', 'NUM_VEH', 'ROAD_COND_CD', 'ROAD_REL_CD', 'ROAD_TYPE_CD',
       'SURF_COND_CD', 'SURF_TYPE_CD', 'WEATHER_CD',
       'PEDESTRIAN', 'DR_SEX', 'From_Louisiana', 'NUM_OCC',
        'city_level', 'highway_type', 'manner_of_collision',
       'aggressive_driving', 'age_group', 'crash_time', 'vehicle_type',
       'drive_distract', 'drive_condition', 'vehicle_year']

int_list = ['CR_MONTH', 'HIT_AND_RUN', 'NUM_VEH','PEDESTRIAN', 'age_group',
            'crash_time', 'NUM_OCC', 'vehicle_year']

df = pd.read_csv('./data_cleaned/' + str(year) + '_cleaned.csv')


Min = df.loc[(df['SEVERITY_CD'] == 'A')]
Middle = df.loc[(df['SEVERITY_CD'] == 'C')| (df['SEVERITY_CD'] == 'B')]
Maj = df.loc[(df['SEVERITY_CD'] != 'A') & (df['SEVERITY_CD'] != 'B') & (df['SEVERITY_CD'] != 'C')]


data_min = Min[features].as_matrix()
label_min = Min['SEVERITY_CD'].values.reshape(-1, 1)
data_Middle = Middle[features].as_matrix()
label_Middle = Middle['SEVERITY_CD'].values.reshape(-1, 1)
data_maj = Maj[features].as_matrix()
label_maj = Maj['SEVERITY_CD'].values.reshape(-1, 1)

data_new = np.append(data_min, data_min, axis=0)
label_new = np.append(label_min, label_min, axis=0)
for i in range(5):
    data_new = np.append(data_new, data_min, axis=0)
    label_new = np.append(label_new, label_min, axis=0)

num_supple = data_Middle.shape[0] - data_new.shape[0]
for i in range(num_supple):
    rand_pick = np.random.randint(data_Middle.shape[0]-1)
    data_new = np.append(data_new, data_Middle[rand_pick, :].reshape(1, -1), axis=0)
    label_new = np.append(label_new, label_Middle[rand_pick, :].reshape(1, -1), axis=0)

data_new = np.append(data_new, data_maj, axis=0)
label_new = np.append(label_new, label_maj, axis=0)

print(iD.count_value(df['SEVERITY_CD'].values))
print(iD.count_value(label_new))

X = pd.DataFrame(data_new, columns=features)
y = pd.DataFrame(label_new, columns=['SEVERITY_CD']).fillna('NaN')
y = LabelEncoder().fit_transform(y)

# ambi = data_min[0, :].reshape(1, data_min[0, :].shape[0])

# for i in range(data_min.shape[0]):
#     print('i', i)
#     for j in range(data_min.shape[0]):
#         if j % 10000 == 0:
#             print('j', j)
#         if np.array_equal(data_min[i, :], data_maj[j, :]):
#             ambi = np.append(ambi, data_min[i, :], axis=0)
#             print(data_min[i, :])
#
# print(ambi.shape)

for col in tqdm(int_list):
    X[str(col)] = X[str(col)].dropna().apply(lambda x: int(x))
    X[str(col)]=X[str(col)].fillna(-1)
    X[str(col)] = LabelEncoder().fit_transform(X[str(col)])
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

X.to_csv('./data_encoded/' + str(year) + '_v2_data.csv', index=False)
y = pd.DataFrame(y)
y.to_csv('./data_encoded/'+str(year)+'_v2_label.csv', index=False)

