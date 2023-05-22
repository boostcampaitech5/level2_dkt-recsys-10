import os

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from pre_process import kfold_preprocess,pre_process,feature_engineering
from train import train
from submission import main

def userid_table(train_df,feature):
    train_df = train_df.sort_values(by=["userID", "Timestamp"], axis=0)
    group = train_df[feature].groupby("userID").apply(lambda x: (
        x["assessmentItemID"].values,
        x["testId"].values,
        x['time_lag'].values,
        x["elapsed"].values,
        x['assessmentItemAverage'].values,
        x['UserAverage'].values,
        x["answerCode"].values))
    return group.values

submission_list = []
auc_array = []
feature = ['userID', 'assessmentItemID', 'testId', 'time_lag', 'answerCode', 
         'elapsed', 'assessmentItemAverage','UserAverage'] 

train_path = "./data/total_data.csv"
train_data = pd.read_csv(train_path)
total_index=train_data['userID'].unique()
print('total_index :',total_index)
# 전처리
train_data = feature_engineering(train_data)
# 유저별 grouping
userby = userid_table(train_data,feature)

test_path = "./data/test_data.csv"
test_data=pd.read_csv(test_path)
# 전처리 후 grouping
pre_process(test_data,0,False)

kf = KFold(n_splits = 5, shuffle = True, random_state = 42)
for idx,(train_idx, valid_idx) in enumerate(kf.split(total_index)):
    print()
    print('kfold_ cv idx : ',idx)
    print('data split and grouping')
    train_df = train_data[train_data['userID'].isin(train_idx)]
    val_df = train_data[train_data['userID'].isin(valid_idx)]
    kfold_preprocess(train_df,val_df,feature)
    
    print('start train')
    best_auc = train()
    auc_array.append(best_auc)
    submissions = main()
    submission_list.append(submissions)

submission_fold = pd.DataFrame()
submission_fold['id'] = np.arange(744)
submission_fold['prediction'] = np.mean(submission_list, axis=0)
submission_fold.to_csv("Saint_kfold.csv", index=False)
print(auc_array)