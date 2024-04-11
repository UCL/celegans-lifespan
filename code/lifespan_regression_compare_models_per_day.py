import numpy as np
import pandas as pd
import os
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import special
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.neural_network import MLPRegressor
from numpy import arange
import xgboost
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, recall_score
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import argparse


NUM_RUNS=20


parser = argparse.ArgumentParser(description='CElegans Lifespan prediction from pathological features')
parser.add_argument('--control_path', dest='control_path', default='/Users/fastmaladmin/Documents/Carina_Worms/new_data/control_data_build_model.xlsx', help='excel file with pathology from non-mutant worms')
parser.add_argument('--mutant_path', dest='mutant_path', default='/Users/fastmaladmin/Documents/Carina_Worms/new_data/mutants_test_model_data.xlsx', help='jexcel file with pathology from mutant worms')
parser.add_argument('--out_dir', dest='out_dir', default='/Users/fastmaladmin/Documents/Carina_Worms/verify_output/output_all_models_average_per_day/', help='path to slide images')

args = parser.parse_args()



#control_path ='/Users/fastmaladmin/Documents/Carina_Worms/new_data/control_data_build_model.xlsx'
#mutant_path='/Users/fastmaladmin/Documents/Carina_Worms/new_data/mutants_test_model_data.xlsx'

out_dir = args.out_dir# '/Users/fastmaladmin/Documents/Carina_Worms/verify_output/output_all_simple_average_per_day'


if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Read excel tables

control_data = pd.read_excel(args.control_path)
control_nonna = control_data.dropna()
mutant_data = pd.read_excel(args.mutant_path)
mutant_nonna = control_data.dropna()
print(control_data.shape)
print(mutant_data.shape)

print(control_data.head)

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

### Remove outliers (Gut)


q = mutant_data["Gut"].quantile(0.90)
#mutant_data=mutant_data[mutant_data['Gut']<q]

# Drop non numerical columns

control_data_no = control_data.drop(columns=['Worm','Type'])
control_array = control_data_no.to_numpy().astype(float)

## Average mutant data per day per worm type

mutant_data = mutant_data.groupby(['Type','Day'], as_index=False).mean()

# Drop non numerical columns

mutant_data_no = mutant_data.drop(columns=['Type'])
mutant_array = mutant_data_no.to_numpy().astype(float)


# Concatenate control and mutant data

all_data_no=pd.concat([control_data_no, mutant_data_no])
all_array=all_data_no.to_numpy().astype(float)

# Select features
X = all_array[:,:-1]
y= all_array[:,-1] - all_array[:,0]

#print(X.shape)
#print(np.any(np.isnan(X)))
#exit()


dict_regressors = {
    "Linreg": LinearRegression(normalize=True),
    "Ridge": Ridge(alpha=100, solver='cholesky', tol=0.0001, random_state=42),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.9, selection='random', random_state=42), #class_weight='balanced'
    "SVM": SVR(kernel = 'rbf'),
    "RF": RandomForestRegressor(n_estimators=500, oob_score=True),#, random_state=100),
    "MLP": MLPRegressor(hidden_layer_sizes=(5,),activation="tanh" ,random_state=1, max_iter=10000),
}

test_dataframe=[]

avg_r2_scores=[]
std_r2_scores=[]

avg_mae_scores=[]
std_mae_scores=[]

avg_rmse_scores=[]
std_rmse_scores=[]

raw_r2_scores=[]
raw_mae_scores=[]
raw_rmse_scores=[]

avg_sensi=[]
std_sensi=[]

avg_speci=[]
std_speci=[]

avg_f1=[]
std_f1=[]

avg_acc=[]
std_acc=[]

raw_sensi_scores=[]
raw_speci_scores=[]

raw_f1_scores=[]
raw_acc_scores=[]

break_down = pd.DataFrame(columns=['Model','Run', 'Day', 'R2', 'MAE'])

THRESH = 18
model_names=[]
for model, model_instantiation in dict_regressors.items():
    r2_scores=[]
    mae_scores=[]
    rmse_scores=[]
    sensitivity_scores=[]
    specificity_scores=[]
    f1_scores=[]
    acc_scores=[]




    model_names.append(model)
    print(model_names)

    for run in range(NUM_RUNS):
        print(model+" run "+ str(run))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        ### Standardize the numerical features
        X_sc = StandardScaler()
        y_sc = StandardScaler()
        y_train_rs = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
        X_train = X_sc.fit_transform(X_train)
        y_train_scaled = y_sc.fit_transform(y_train_rs)
        model_instantiation.fit(X_train, y_train_scaled)
        y_pred = model_instantiation.predict(X_sc.transform(X_test))
        y_pred = y_pred.reshape(-1, 1)
        y_pred = y_sc.inverse_transform(y_pred)
        r2_scores.append(r2_score(y_test, y_pred))
        mae_scores.append(mean_absolute_error(y_test, y_pred))
        rmse_scores.append(sqrt(mean_absolute_error(y_test, y_pred)))


        tps = np.sum((y_pred>=THRESH)*(y_test>=THRESH)).astype(float)
        fns = np.sum((y_pred<THRESH)*(y_test>=THRESH)).astype(float)
        fps = np.sum((y_pred>=THRESH)*(y_test<THRESH)).astype(float)
        tns = np.sum((y_pred<THRESH)*(y_test<THRESH)).astype(float)

        #sensitivity_scores.append(tps/(tps+fns))
        #specificity_scores.append(tns/(tns+fps))
        sensitivity_scores.append(recall_score(y_test>=THRESH, y_pred>=THRESH))
        specificity_scores.append(recall_score(y_test >= THRESH, y_pred >= THRESH, pos_label=0))
        f1_scores.append(f1_score(y_test>=THRESH, y_pred>=THRESH))
        acc_scores.append(accuracy_score(y_test>=THRESH, y_pred>=THRESH))


        ####### per day evaluation #######
        days=X_test[:,0]

        print(X_test[:,0])
        res = pd.DataFrame({'Predicted Life Span': y_pred.ravel()+days.ravel(), 'Real Life Span': y_test.ravel()+days.ravel(), 'Day': X_test[:,0]})

        for day in np.unique(days):
            day_results = res.loc[res['Day'] == day]
            day_r2 = r2_score(day_results['Real Life Span'], day_results['Predicted Life Span'])
            day_mae = mean_absolute_error(day_results['Real Life Span'], day_results['Predicted Life Span'])
            day_recall = recall_score(day_results['Real Life Span']>=THRESH, day_results['Predicted Life Span']>=THRESH)
            day_specificity = recall_score(day_results['Real Life Span'] >= THRESH,
                                      day_results['Predicted Life Span'] >= THRESH, pos_label=0)

            day_f1score = f1_score(day_results['Real Life Span'] >= THRESH,
                                      day_results['Predicted Life Span'] >= THRESH)
            day_acc = accuracy_score(day_results['Real Life Span'] >= THRESH,
                                      day_results['Predicted Life Span'] >= THRESH)
            break_down = break_down.append({'Model':model,'Run':run, 'Day': day, 'R2': day_r2, 'MAE': day_mae, 'Recall':day_recall,
                                            'Specificity':day_specificity, 'F1':day_f1score, 'Acc': day_acc}, ignore_index=True)




        ##################################


    avg_r2_scores.append(np.mean(r2_scores))
    std_r2_scores.append(np.std(r2_scores))
    avg_mae_scores.append(np.mean(mae_scores))
    std_mae_scores.append(np.std(mae_scores))
    avg_rmse_scores.append(np.mean(rmse_scores))
    std_rmse_scores.append(np.std(rmse_scores))

    avg_sensi.append(np.mean(sensitivity_scores))
    std_sensi.append(np.std(sensitivity_scores))

    avg_speci.append(np.mean(specificity_scores))
    std_speci.append(np.std(specificity_scores))

    avg_f1.append(np.mean(f1_scores))
    std_f1.append(np.std(f1_scores))

    avg_acc.append(np.mean(acc_scores))
    std_acc.append(np.std(acc_scores))

    raw_r2_scores.append(r2_scores)
    raw_mae_scores.append(mae_scores)
    raw_rmse_scores.append(rmse_scores)
    raw_sensi_scores.append(sensitivity_scores)
    raw_speci_scores.append(specificity_scores)
    raw_f1_scores.append(f1_scores)
    raw_acc_scores.append(acc_scores)


# store everything in a panda dataframe

res_df=pd.DataFrame()
res_df['Model']=model_names
res_df['R2 mean']=avg_r2_scores
res_df['R2 std']=std_r2_scores
res_df['MAE mean']=avg_mae_scores
res_df['MAE std']=std_mae_scores
res_df['RMSE mean']=avg_rmse_scores
res_df['RMSE std']=std_rmse_scores

res_df['Sensitivity mean']=avg_sensi
res_df['Sensitivity std']=std_sensi

res_df['Specificity mean']=avg_speci
res_df['Specificity std']=std_speci

res_df['F1 mean']=avg_f1
res_df['F1 std']=std_f1

res_df['Acc mean'] = avg_acc
res_df['Acc std'] = std_acc


print(res_df)

#exit()

#pd_r2_scores=pd.Series(raw_r2_scores, index=model_names)

pd_r2_scores=pd.DataFrame(np.transpose(np.array(raw_r2_scores)), columns=model_names)

##### Plots

pd_r2_scores.to_excel(os.path.join(out_dir, 'r2_scores_model_comparison_simple.xlsx'))

##### Save to excel
pd_mae_scores=pd.DataFrame(np.transpose(np.array(raw_mae_scores)), columns=model_names)
pd_mae_scores.to_excel(os.path.join(out_dir, 'mae_scores_model_comparison_simple.xlsx'))

##### Save to excel
pd_rmse_scores=pd.DataFrame(np.transpose(np.array(raw_rmse_scores)), columns=model_names)
pd_rmse_scores.to_excel(os.path.join(out_dir, 'rmse_scores_model_comparison_simple.xlsx'))

pd_f1_scores=pd.DataFrame(np.transpose(np.array(raw_f1_scores)), columns=model_names)
pd_f1_scores.to_excel(os.path.join(out_dir, 'f1_scores_model_comparison_simple.xlsx'))

pd_acc_scores = pd.DataFrame(np.transpose(np.array(raw_acc_scores)), columns=model_names)
pd_acc_scores.to_excel(os.path.join(out_dir, 'acc_scores_model_comparison_simple.xlsx'))

pd_sensitivity_scores=pd.DataFrame(np.transpose(np.array(raw_sensi_scores)), columns=model_names)
pd_sensitivity_scores.to_excel(os.path.join(out_dir, 'sensitivity_model_comparison_simple.xlsx'))

pd_specificity_scores=pd.DataFrame(np.transpose(np.array(raw_speci_scores)), columns=model_names)
pd_specificity_scores.to_excel(os.path.join(out_dir, 'specificity_model_comparison_simple.xlsx'))

fig=plt.figure()
ax= sns.boxplot(
    data=raw_r2_scores,
    #palette=[sns.xkcd_rgb["pale red"], sns.xkcd_rgb["medium green"]],
    showmeans=True,
)
ax.set_xticklabels(model_names)
ax.set_ylabel("r2 score")
#plt.show()

plt.savefig(os.path.join(out_dir, 'r2scores_all.png'), bbox_inches='tight')
fig=plt.figure()
ax= sns.boxplot(
    data=raw_mae_scores,
    #palette=[sns.xkcd_rgb["pale red"], sns.xkcd_rgb["medium green"]],
    showmeans=True,
    #y="r2 score"
)
ax.set_xticklabels(model_names)
ax.set_ylabel("mean average error")
#plt.show()

plt.savefig(os.path.join(out_dir, 'mae_scores_all.png'), bbox_inches='tight')
fig=plt.figure()
ax= sns.boxplot(
    data=raw_rmse_scores,
    #palette=[sns.xkcd_rgb["pale red"], sns.xkcd_rgb["medium green"]],
    showmeans=True,
    #y="r2 score"
)
ax.set_xticklabels(model_names)
ax.set_ylabel("root mean squared error")
#plt.show()

plt.savefig(os.path.join(out_dir, 'rmse_scores_all.png'), bbox_inches='tight')


break_down.to_excel(os.path.join(out_dir, 'r2_scores_model_comparison_simple_per_day.xlsx'))

############################################################
###########################################################

fig=plt.figure()
ax= sns.boxplot(
    data=raw_sensi_scores,
    #palette=[sns.xkcd_rgb["pale red"], sns.xkcd_rgb["medium green"]],
    showmeans=True,
    #y="r2 score"
)
ax.set_xticklabels(model_names)
ax.set_ylabel("Sensitivity")
#plt.show()

plt.savefig(os.path.join(out_dir, 'sensitivity_scores_all.png'), bbox_inches='tight')


################################################################
####################################################

fig=plt.figure()
ax= sns.boxplot(
    data=raw_speci_scores,
    #palette=[sns.xkcd_rgb["pale red"], sns.xkcd_rgb["medium green"]],
    showmeans=True,
    #y="r2 score"
)
ax.set_xticklabels(model_names)
ax.set_ylabel("Specificity")
#plt.show()

plt.savefig(os.path.join(out_dir, 'specificity_scores_all.png'), bbox_inches='tight')

#################################################
###################################################


fig=plt.figure()
ax= sns.boxplot(
    data=raw_f1_scores,
    #palette=[sns.xkcd_rgb["pale red"], sns.xkcd_rgb["medium green"]],
    showmeans=True,
    #y="r2 score"
)
ax.set_xticklabels(model_names)
ax.set_ylabel("F1 score")
#plt.show()

plt.savefig(os.path.join(out_dir, 'f1score_scores_all.png'), bbox_inches='tight')

fig = plt.figure()
ax = sns.boxplot(
    data=raw_acc_scores,
    # palette=[sns.xkcd_rgb["pale red"], sns.xkcd_rgb["medium green"]],
    showmeans=True,
    # y="r2 score"
)
ax.set_xticklabels(model_names)
ax.set_ylabel("Acc score")
# plt.show()

plt.savefig(os.path.join(out_dir, 'acc_scores_all.png'), bbox_inches='tight')
#####################################
#####################################


for day in np.unique(X[:,0]):
    day_results = break_down.loc[break_down['Day'] == day]
    print(day_results)
    #exit()
    r2_day_results_list=[]

    mae_day_results_list = []

    specificity_day_results_list=[]
    sensitivity_day_results_list=[]
    f1_day_results_list = []
    acc_day_results_list=[]


    for model in model_names:
        model_day_results = day_results.loc[day_results['Model'] == model]
        r2_day_results_list.append(model_day_results['R2'])
        mae_day_results_list.append(model_day_results['MAE'])
        specificity_day_results_list.append(model_day_results['Specificity'])
        sensitivity_day_results_list.append(model_day_results['Recall'])
        f1_day_results_list.append(model_day_results['F1'])
        acc_day_results_list.append(model_day_results['Acc'])






    fig = plt.figure()
    ax = sns.boxplot(
        data=r2_day_results_list,
        # palette=[sns.xkcd_rgb["pale red"], sns.xkcd_rgb["medium green"]],
        showmeans=True,
        # y="r2 score"
    )
    ax.set_xticklabels(model_names)
    ax.set_ylabel("r2 score")
    ax.set_title('Day: '+str(day))

    plt.savefig(os.path.join(out_dir, 'day'+str(day)+'r2_scores_all.png'), bbox_inches='tight')

    fig = plt.figure()
    ax = sns.boxplot(
        data=mae_day_results_list,
        # palette=[sns.xkcd_rgb["pale red"], sns.xkcd_rgb["medium green"]],
        showmeans=True,
        # y="r2 score"
    )
    ax.set_xticklabels(model_names)
    ax.set_ylabel("mean absolute error")
    ax.set_title('Day: '+str(day))

    plt.savefig(os.path.join(out_dir, 'day'+str(day)+'mae_scores_all.png'), bbox_inches='tight')




    fig = plt.figure()
    ax = sns.boxplot(
        data= specificity_day_results_list,
        # palette=[sns.xkcd_rgb["pale red"], sns.xkcd_rgb["medium green"]],
        showmeans=True,
        # y="r2 score"
    )
    ax.set_xticklabels(model_names)
    ax.set_ylabel("Specificity")
    ax.set_title('Day: '+str(day))

    plt.savefig(os.path.join(out_dir, 'day'+str(day)+'specificity_scores_all.png'), bbox_inches='tight')


    fig = plt.figure()
    ax = sns.boxplot(
        data= sensitivity_day_results_list,
        # palette=[sns.xkcd_rgb["pale red"], sns.xkcd_rgb["medium green"]],
        showmeans=True,
        # y="r2 score"
    )
    ax.set_xticklabels(model_names)
    ax.set_ylabel("Sensitivity")
    ax.set_title('Day: '+str(day))

    plt.savefig(os.path.join(out_dir, 'day'+str(day)+'sensitivity_scores_all.png'), bbox_inches='tight')


    fig = plt.figure()
    ax = sns.boxplot(
        data= f1_day_results_list,
        # palette=[sns.xkcd_rgb["pale red"], sns.xkcd_rgb["medium green"]],
        showmeans=True,
        # y="r2 score"
    )
    ax.set_xticklabels(model_names)
    ax.set_ylabel("F1")
    ax.set_title('Day: '+str(day))

    plt.savefig(os.path.join(out_dir, 'day'+str(day)+'f1_scores_all.png'), bbox_inches='tight')


    fig = plt.figure()
    ax = sns.boxplot(
        data= acc_day_results_list,
        # palette=[sns.xkcd_rgb["pale red"], sns.xkcd_rgb["medium green"]],
        showmeans=True,
        # y="r2 score"
    )
    ax.set_xticklabels(model_names)
    ax.set_ylabel("Acc")
    ax.set_title('Day: '+str(day))

    plt.savefig(os.path.join(out_dir, 'day'+str(day)+'acc_scores_all.png'), bbox_inches='tight')