import time
from datetime import datetime
import pandas as pd
import numpy as np
from collections import defaultdict

# 개인별 문제 푸는데 걸린 시간
def get_time_lag(df):
    time_dict = {}
    time_lag = np.zeros(len(df), dtype=np.float32)
    for idx, row in enumerate(df[["userID", "Timestamp", "testId"]].values):
        row[1] = time.mktime(datetime.strptime(row[1],'%Y-%m-%d %H:%M:%S').timetuple())
        # 처음 문제 푸는 유저
        if row[0] not in time_dict:
            time_lag[idx] = 0
            time_dict[row[0]] = [row[1], row[2], 0] # last_timestamp, last_task_container_id, last_lagtime
        
        else:
            # 이 시험지를 풀어봤다면
            if row[2] == time_dict[row[0]][1]:
                time_lag[idx] = time_dict[row[0]][2]
            # 이 시험지를 푼 적 없다면
            else:
                time_lag[idx] = row[1] - time_dict[row[0]][0]
                time_dict[row[0]][0] = row[1]
                time_dict[row[0]][1] = row[2]
                time_dict[row[0]][2] = time_lag[idx]

    df["time_lag"] = time_lag/1000/60 # convert to miniute
    # 문제푼지 하루가 지났다면 1440(60*24)로 만들어주고, 아니라면 그대로, 0보다 작다면 0으로 만들어줌
    df["time_lag"] = df["time_lag"].clip(0, 1440) 
    # return time_dict
    return df

# 문제 푸는데 걸린 시간
def duration(df):
    df = df.sort_values(by=['userID', 'Timestamp']).reset_index(drop=True)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['months'] = df['Timestamp'].dt.month
    df['days'] = df['Timestamp'].dt.day
    df['ts'] = df['Timestamp'].map(pd.Timestamp.timestamp)
    df['prev_ts'] = df.groupby(['userID', 'testId', 'months','days'])['ts'].shift(1)
    df["prev_ts"] = df["prev_ts"].fillna(0)
    df["elapsed"] = np.where(df["prev_ts"] == 0, 0, df["ts"] - df["prev_ts"])

    indexes = df[df['elapsed'] > 1200].index
    df.loc[indexes, 'elapsed'] = 1200
    df = df.drop(['months','days','ts','prev_ts'],axis='columns')
    return df

# 문항별 평균
def make_assess_ratio(df):
    ratio_dict = defaultdict(float)
    grouped_dict = dict(df.groupby('assessmentItemID')['answerCode'].value_counts())
    assess_keys = list(set([x[0] for x in grouped_dict.keys()]))
    for key in assess_keys:
        if grouped_dict.get((key, 1)):
            right = grouped_dict[(key, 1)]
        else:
            right=0
        if grouped_dict.get((key, 0)):
            wrong = grouped_dict[(key, 0)]
        else:
            wrong = 0
        ratio = right / (right + wrong)
        ratio_dict[key] = ratio

    df['assessmentItemAverage'] = df['assessmentItemID'].map(ratio_dict)
    return df

# 유저별 평균
def make_user_ratio(df):
    ratio_dict = defaultdict(float)
    grouped_dict = dict(df.groupby('userID')['answerCode'].value_counts())
    user_keys = list(set([x[0] for x in grouped_dict.keys()]))
    for key in user_keys:
        if grouped_dict.get((key, 1)):
            right = grouped_dict[(key, 1)]
        else:
            right = 0
        if grouped_dict.get((key, 0)):
            wrong = grouped_dict[(key, 0)]
        else:
            wrong = 0
        ratio = right / (right + wrong)
        ratio_dict[key] = ratio

    df['UserAverage'] = df['userID'].map(ratio_dict)
    return df