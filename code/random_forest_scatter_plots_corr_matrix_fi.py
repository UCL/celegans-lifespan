import numpy as np
import pandas as pd
import os
import csv
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import special
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import argparse


# Read excel tables
parser = argparse.ArgumentParser(description='CElegans Lifespan prediction from pathological features')
parser.add_argument('--control_path', dest='control_path', default='/Users/fastmaladmin/Documents/Carina_Worms/new_data/control_data_build_model.xlsx', help='excel file with pathology from non-mutant worms')
parser.add_argument('--mutant_path', dest='mutant_path', default='/Users/fastmaladmin/Documents/Carina_Worms/new_data/mutants_test_model_data.xlsx', help='jexcel file with pathology from mutant worms')
parser.add_argument('--out_dir', dest='out_dir', default='/Users/fastmaladmin/Documents/Carina_Worms/verify_output/output_all_models_average_per_day/', help='path to slide images')

args = parser.parse_args()

out_dir=args.out_dir
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

control_data = pd.read_excel(args.control_path)
control_nonna = control_data.dropna()

mutant_data = pd.read_excel(args.mutant_path)
mutant_nonna = mutant_data.dropna()


print('Initial Control Data: ', control_data.shape)
print('Initial Mutant Data: ', mutant_data.shape)

##############################################
################ FILL MISSING VALUES #########

# Missing values are filled with the day average of existing values

for day in [1,4,7,9,11,14]:
    day_mean = control_nonna.loc[control_nonna['Day'] == day].mean()
    for feature in ["Pharynx", "Gonad", "Tumor", "Yolk", "Gut", "ObvLifespan"]:
        control_data[feature] = np.where((control_data['Day'] == day)&(control_data[feature].isna()), day_mean[feature], control_data[feature])
print(sum(control_data.loc[control_data['Day']==7]["Pharynx"].isna()))

for day in [1,4,7,9,11,14]:
    day_mean = mutant_nonna.loc[mutant_nonna['Day'] == day].mean()
    for feature in ["Pharynx", "Gonad", "Tumor", "Yolk", "Gut", "ObvLifespan"]:
        mutant_data[feature] = np.where((mutant_data['Day'] == day)&(mutant_data[feature].isna()), day_mean[feature], mutant_data[feature])

mutant_data = mutant_data.dropna()


# Drop non numerical columns

control_data_no = control_data.drop(columns=['Worm','Type'])


# Add a numerical value corresponding to the type of worm (control =0; mutant =1); this is for plotting purposes

control_data_no.insert(0, "Mutant",0)
print(control_data_no)

control_array = control_data_no.to_numpy().astype(float)


### Remove outliers (Gut)
q = mutant_data["Gut"].quantile(0.90)
mutant_data=mutant_data[mutant_data['Gut']<q]

## Average mutant data per day per worm type

mutant_data = mutant_data.groupby(['Type','Day'], as_index=False).mean()

# Remove non numerical values

mutant_data_no = mutant_data.drop(columns=['Type'])

# Add a numerical value corresponding to the type
mutant_data_no.insert(0, "Mutant",1)



#concatenat mutant and control data

all_data_no=pd.concat([control_data_no, mutant_data_no])


corr = all_data_no.iloc[:, 2:].corr()
sns.heatmap(corr, cmap="Reds", annot=True)
plt.show()

all_array=all_data_no.to_numpy().astype(float)


# Select features
X = all_array[:,:-1]

# Target value to predict : ObvLifeSpan - Day (obseration)
y= all_array[:,-1] - all_array[:,1]



#x = all_data_no.values #returns a numpy array
#min_max_scaler = StandardScaler()
#x_scaled = min_max_scaler.fit_transform(x)
#df = pd.DataFrame(x_scaled)

df=all_data_no.copy()
df.loc[:,'Yolk'] *= -1
df.loc[:,'Tumor'] *= -1
df.loc[:,'Gonad'] *= -1
df.loc[:,'Pharynx'] *= -1
corr2 = df.iloc[:, 2:].corr()
sns.heatmap(corr2, cmap="Reds", annot=True)
plt.show()



# Spliting the datdset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.2, random_state=24)



### Standardize the numerical features


X_sc = StandardScaler()
y_sc = StandardScaler()
y_train_rs=y_train.reshape(-1,1)
y_test=y_test.reshape(-1,1)
X_train[:,1:] = X_sc.fit_transform(X_train[:,1:])
y_train_scaled =y_sc.fit_transform(y_train_rs)



############ Build a ML regression model (Random Forest)

#from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from numpy import arange
import xgboost

print(X_train.shape, X_test.shape)

regrassor =  RandomForestRegressor(n_estimators=500, oob_score=True, random_state=100)

regrassor.fit(X_train[:,1:], y_train_scaled)

y_pred = regrassor.predict(X_sc.transform(X_test[:,1:]))
y_pred=y_pred.reshape(-1,1)
y_pred = y_sc.inverse_transform(y_pred)
days=X_test[:,1]
train_days = X_train[:,1]
num_type=X_test[:,0]
type=[]
for xx in range(len(num_type)):
    if num_type[xx]==0:
        type.append('Control')
    else:
        type.append('Mutant')

res = pd.DataFrame({'Predicted Life Span': y_pred.ravel()+days.ravel(), 'Real Life Span': y_test.ravel()+days.ravel(), 'Day': days, 'Type':type})
print(res)
import math
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
print(r2_score(y_test, y_pred))
print(mean_absolute_error(y_test, y_pred))


#### Compute the scores by day

res.sort_values('Day')
brake_down=pd.DataFrame(columns=['Day', 'R2', 'MAE'])
legend_labels=[]
for day in np.unique(days):
    day_results = res.loc[res['Day']==day]
    day_r2=r2_score(day_results['Real Life Span'], day_results['Predicted Life Span'])
    day_mae=mean_absolute_error(day_results['Real Life Span'], day_results['Predicted Life Span'])
    #print([day, day_r2, day_mae])
    brake_down=brake_down.append({'Day':day, 'R2':day_r2, 'MAE':day_mae}, ignore_index=True)
    legend_labels.append('Day: '+str(int(day))+', R2:'+str(round(day_r2,2))+', MAE:'+str(round(day_mae,2)))
print(brake_down)



##### PLOT SCATTERPLOT #########

fig = plt.figure()
ax = fig.add_subplot(111)
sns.scatterplot(data=res, x='Real Life Span', y='Predicted Life Span', hue='Day', palette='Reds')
ax.set_aspect('equal')
#ax.set_xlim(0, 40)
#ax.set_ylim(0, 40)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[0:], labels=legend_labels)

plt.show()

#### Compute the scores by type (control vs mutant)

brake_down=pd.DataFrame(columns=['Type', 'R2', 'MAE'])
legend_labels=[]
for day in np.unique(res['Type']):
    day_results = res.loc[res['Type']==day]
    day_r2=r2_score(day_results['Real Life Span'], day_results['Predicted Life Span'])
    day_mae=mean_absolute_error(day_results['Real Life Span'], day_results['Predicted Life Span'])
    #print([day, day_r2, day_mae])
    brake_down=brake_down.append({'Type':day, 'R2':day_r2, 'MAE':day_mae}, ignore_index=True)
    #res.loc[res['Day']==day, 'Day'] = 'Day: '+str(int(day))+', R2:'+str(round(day_r2,2))+', MAE:'+str(round(day_mae,2))
    legend_labels.append(day+', R2:'+str(round(day_r2,2))+', MAE:'+str(round(day_mae,2)))
print(brake_down)

##### PLOT SCATTERPLOT #########


fig = plt.figure()
ax = fig.add_subplot(111)
sns.scatterplot(data=res, x='Real Life Span', y='Predicted Life Span', hue='Type', palette='icefire')
ax.set_aspect('equal')
#ax.set_xlim(-5, 40)
#ax.set_ylim(-5, 40)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[0:], labels=reversed(legend_labels))
plt.show()


################### COMPUTE FEATURE IMPORTANCE ###########################
print(y_test.shape)
high_rate_train = y_train.ravel()+train_days.ravel()>20
high_rate_test = y_test.ravel()+days.ravel()>20

low_rate_train = y_train.ravel()+train_days.ravel()<20
low_rate_test = y_test.ravel()+days.ravel()<20
#print(high_rate)


feature_names = list(all_data_no.columns[1:-1].values)

from rfpimp import *

#exit()




##### 1. Using MDI #########

feature_names = list(all_data_no.columns[1:-1].values)
importances = regrassor.feature_importances_
std = np.std([tree.feature_importances_ for tree in regrassor.estimators_], axis=0)
forest_importances = pd.Series(importances, index=feature_names)

fig, ax = plt.subplots()
forest_importances.plot.barh(yerr=std, ax=ax, color='purple')
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
plt.show()

forest_importances.to_excel(os.path.join(out_dir, 'rf_importances_mdi.xlsx'))



############################################
############################### Feature Importances for low and high survival rates

high_rate_train = y_train.ravel()+train_days.ravel()>20
high_rate_test = y_test.ravel()+days.ravel()>20

low_rate_train = y_train.ravel()+train_days.ravel()<20
low_rate_test = y_test.ravel()+days.ravel()<20
#print(high_rate)


feature_names = list(all_data_no.columns[1:-1].values)



###################################################################################
####################################################################################
regrassor_high =  RandomForestRegressor(n_estimators=500, oob_score=True, random_state=100)
regrassor_high.fit(X_train[high_rate_train,1:], y_train_scaled[high_rate_train])
importances_high = regrassor_high.feature_importances_
std = np.std([tree.feature_importances_ for tree in regrassor_high.estimators_], axis=0)
forest_importances = pd.Series(importances_high, index=feature_names)
fig, ax = plt.subplots()
forest_importances.plot.barh(yerr=std, ax=ax, color='purple')
ax.set_title("Feature importances using MDI High")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
plt.show()


forest_importances.to_excel(os.path.join(out_dir, 'rf_importances_mdi_high.xlsx'))
########################################################################################
#####################################################################################

###################################################################################
####################################################################################
regrassor_low =  RandomForestRegressor(n_estimators=500, oob_score=True, random_state=100)
regrassor_low.fit(X_train[low_rate_train,1:], y_train_scaled[low_rate_train])
importances_low = regrassor_low.feature_importances_
std = np.std([tree.feature_importances_ for tree in regrassor_low.estimators_], axis=0)
forest_importances = pd.Series(importances_low, index=feature_names)
fig, ax = plt.subplots()
forest_importances.plot.barh(yerr=std, ax=ax, color='purple')
ax.set_title("Feature importances using MDI Low")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
plt.show()


forest_importances.to_excel(os.path.join(out_dir, 'rf_importances_mdi_low.xlsx'))
########################################################################################
#####################################################################################




#### 2. Using PI


#from sklearn.inspection import permutation_importance
#from rfpimp import *
#
# from sklearn.inspection import permutation_importance
#
# result = permutation_importance(
#     regrassor, X_sc.transform(X_test[:,1:]), y_sc.fit_transform(y_test), n_repeats=10, random_state=42, n_jobs=2
# )
#
# sorted_importances_idx = result.importances_mean.argsort()
# importances = pd.DataFrame(
#     #result.importances[sorted_importances_idx].T,
#     result.importances.T,
#     columns=feature_names,
# )
# ax = importances.plot.box(vert=False, whis=10)
# ax.set_title("Permutation Importances (test set)")
# ax.axvline(x=0, color="k", linestyle="--")
# ax.set_xlabel("Decrease in accuracy score")
# ax.figure.tight_layout()
# plt.show()
#
# importances.to_excel(os.path.join(out_dir, 'rf_importances_pi_test.xlsx'))
#
# result = permutation_importance(
#     regrassor, X_train[:,1:], y_train_scaled, n_repeats=10, random_state=42, n_jobs=2
# )
#
# sorted_importances_idx = result.importances_mean.argsort()
# importances = pd.DataFrame(
#     #result.importances[sorted_importances_idx].T,
#     result.importances.T,
#     columns=feature_names,
# )
# ax = importances.plot.box(vert=False, whis=10)
# ax.set_title("Permutation Importances (train set)")
# ax.axvline(x=0, color="k", linestyle="--")
# ax.set_xlabel("Decrease in accuracy score")
# ax.figure.tight_layout()
# plt.show()
#
# importances.to_excel(os.path.join(out_dir, 'rf_importances_pi_train.xlsx'))
#
#
# import shap
# explainer = shap.TreeExplainer(regrassor)
# shap_values = explainer.shap_values(pd.DataFrame(X_test[:,1:], columns=feature_names))
# shap.summary_plot(shap_values, pd.DataFrame(X_test[:,1:], columns=feature_names), plot_type="bar")