import time
import pickle
import pandas as pd
from utils import get_time_lag,duration,make_assess_ratio,make_user_ratio

# 값을 0부터 시작으로 바꾸기
def indexing(tag,df):
    tag_indexing = {v:k for k,v in enumerate(df[tag].unique())}
    df[tag] = df[tag].map(tag_indexing)
    return df

# 데이터 전처리
def feature_engineering(train_df):
    print("Start feature-engineering")
    t_s = time.time()
    train_df.index = train_df.index.astype('uint32')
    
    # get time_lag feature
    train_df = get_time_lag(train_df)
    # with open("time_dict95.pkl.zip", 'wb') as pick:
    #     pickle.dump(time_dict, pick)
    
    # 문제 푼 시간
    train_df = duration(train_df)
    # 문제별 평균
    train_df = make_assess_ratio(train_df)
    # 유저별 평균
    train_df = make_user_ratio(train_df)
    # 문제번호 1부터 시작으로 바꾸기
    train_df = indexing('assessmentItemID',train_df)
    train_df["assessmentItemID"] += 1
    # 시험지번호 1부터 시작으로 바꾸기
    train_df = indexing('testId',train_df)
    train_df["testId"] += 1
    # 문제 맞추면 2, 틀리면 1로 바꾸기
    train_df["answerCode"] += 1
    print("Complete feature-engineering, execution time {:.2f} s".format(time.time() - t_s))
    return train_df

def grouping(df,features,pickle_name):
    df_group = df[features].groupby("userID").apply(lambda df: (
            df["assessmentItemID"].values,
            df["testId"].values,
            df['time_lag'].values,
            df["elapsed"].values, 
            df['assessmentItemAverage'].values,
            df['UserAverage'].values,
            df["answerCode"].values
        ))
    with open(f"./after_data/{pickle_name}.pkl.zip", 'wb') as pick:
        pickle.dump(df_group, pick)
        
def kfold_preprocess(train_df,val_df,features):
    t_s = time.time()

    print("Start train and Val grouping")
    grouping(train_df,features,'train_group90')
    grouping(val_df,features,'val_group90')
    print("Complete Pre-processing, execution time {:.2f} s".format(time.time() - t_s))
    
# train == True : df 데이터를 전처리 후 split_raito 비율로 train과 valid로 나누고 grouping
# train == False : df를 전처리 후 grouping
def pre_process(df, split_ratio=0.9,train=True):
    print("Start pre-process")
    t_s = time.time()
    # 전처리
    df=feature_engineering(df)
    
    features = ['userID', 'assessmentItemID', 'testId', 
                'time_lag', 'elapsed', # 시간 관련 
                'UserAverage', 'assessmentItemAverage', # 유저평균, 문제평균
                'answerCode',]
    
    # 학습데이터
    if train:
        # 데이터 나누기 
        num_rows = df.shape[0]
        print(f'{num_rows} * {split_ratio} = {num_rows * split_ratio}')
        val_df = df[int(num_rows * split_ratio):]
        train_df = df[:int(num_rows * split_ratio)]

        print("Train dataframe shape after process ({}, {})/ Val dataframe shape after process({}, {})".format(
            train_df.shape[0], train_df.shape[1], val_df.shape[0], val_df.shape[1]))
        print("====================")

        print("Start train and Val grouping")
        grouping(train_df,features,'train_group90')
        grouping(val_df,features,'val_group90')
        print("Complete pre-process, execution time {:.2f} s".format(time.time() - t_s))
    
    # 평가 데이터
    else:
        print("Start test grouping")
        grouping(df,features,'test_group90')
        print("Complete pre-process, execution time {:.2f} s".format(time.time() - t_s))


if __name__ == "__main__":
    # train과 test 합쳐서 answerCode가 -1인 것 제외
    # total_data_path = './data/total_data.csv'
    # total_df = pd.read_csv(total_data_path)
    # pre_process(total_df,0.9,True)
    
    # train_data_path = './data/train_data.csv'
    # train_df = pd.read_csv(train_data_path)
    # pre_process(train_df, 0.90, True)
    
    test_path = './data/test_data.csv'
    test_df=pd.read_csv(test_path)
    pre_process(test_df, 0, False)
