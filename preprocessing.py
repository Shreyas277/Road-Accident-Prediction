from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import accuracy_score, mean_squared_error , r2_score 
import matplotlib.pyplot as plt
from lightgbm import early_stopping, log_evaluation

df = pd.read_csv("train.csv")

target_column = 'accident_risk'
df=df.drop(columns='id')
y = df[target_column]
df['holiday']=df['holiday'].astype(int)
df['road_signs_present']=df['road_signs_present'].astype(int)
df['public_road']=df['public_road'].astype(int)
df['school_season']=df['school_season'].astype(int)
df['lighting'] = df['lighting'].astype('category')
df['road_type'] = df['road_type'].astype('category')
df['weather'] = df['weather'].astype('category')
df['time_of_day'] = df['time_of_day'].astype('category')
X = df.drop(columns=[target_column])
print(X)


