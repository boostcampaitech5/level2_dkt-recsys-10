import pandas as pd

# def train_test_split(dataset, config):
def train_test_split(dataset, submission_dataframe, config):

    print('-------- train test split --------')
    train_data_cut = config.dataset_feature.train_data_cut
    test_split = config.dataset_feature.test_split
    keys = list(config.dataset_feature.valid_feature)
    
    label = config.dataset_feature.label
    if train_data_cut == 'all':
        test_dataset_tail = dataset.groupby('userID').apply(lambda x: x.tail(int(len(x) * test_split))).reset_index(drop=True)
        test_dataset_head = dataset.groupby('userID').apply(lambda x: x.head(int(len(x) * test_split))).reset_index(drop=True)
        test_dataset = pd.concat([test_dataset_head, test_dataset_tail], ignore_index=True)

    elif train_data_cut == 'head':
        test_dataset = dataset.groupby('userID').apply(lambda x: x.head(int(len(x) * test_split))).reset_index(drop=True)

    elif train_data_cut == 'tail':
        test_dataset = dataset.groupby('userID').apply(lambda x: x.tail(int(len(x) * test_split))).reset_index(drop=True)
        
    elif train_data_cut == 'head_tail_one':
        test_dataset_head = dataset[dataset["userID"] != dataset["userID"].shift(1)]
        test_dataset_tail = dataset[dataset["userID"] != dataset["userID"].shift(-1)]
        test_dataset = pd.concat([test_dataset_head, test_dataset_tail], ignore_index=True)
    
    elif train_data_cut == 'head_tail_group_one':
        test_dataset_head = dataset.groupby(["userID", "testId"]).head(1).reset_index(drop=True)
        test_dataset_tail = dataset.groupby(["userID", "testId"]).tail(1).reset_index(drop=True)
        test_dataset = pd.concat([test_dataset_head, test_dataset_tail], ignore_index=True)

    elif train_data_cut == 'head_one':
        test_dataset = dataset[dataset["userID"] != dataset["userID"].shift(1)]

    elif train_data_cut == 'tail_one':
        test_dataset = dataset[dataset["userID"] != dataset["userID"].shift(-1)]

        
    X_train, y_train, X_test, y_test = _split_process(dataset, keys, label, test_dataset)

    # drop
    X_train = X_train.drop(config.catboost_setting.drop_columns, axis=1)
    X_test = X_test.drop(config.catboost_setting.drop_columns, axis=1)
    submission_dataframe = submission_dataframe.drop(config.catboost_setting.drop_columns, axis=1)
    print('--------- split success ----------')
    return X_train, y_train, X_test, y_test, submission_dataframe

def _split_process(dataset, keys, label, test_dataset):
    train_dataset = dataset.merge(test_dataset[keys], on=keys, how='left', indicator=True)
    train_dataset = train_dataset[train_dataset['_merge'] == 'left_only']
    train_dataset = train_dataset.drop('_merge', axis=1).reset_index(drop=True)

    X_train, y_train = train_dataset.drop([label], axis=1), train_dataset[label]
    X_test, y_test = test_dataset.drop([label], axis=1), test_dataset[label]

    return X_train, y_train, X_test, y_test