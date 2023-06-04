import pandas as pd
import omegaconf

from .data_loader import dataloader_decorator


@dataloader_decorator
def data_process(dataset_data: pd.DataFrame, submission_data: pd.DataFrame,
                 config: omegaconf.DictConfig)->pd.DataFrame:
    print('----------- dataprocess -----------')
    dataset_data = _add_data_process(dataset_data, config)
    submission_data = _add_data_process(submission_data, config)

    # int, float를 제외한 값 제거
    dataset_data = dataset_data.select_dtypes(include=['int', 'float'])
    submission_data = submission_data.select_dtypes(include=['int', 'float'])
    print('--------- process success ---------')
    return dataset_data, submission_data


# 데이터 수동 전처리
def _add_data_process(df: pd.DataFrame, config: omegaconf.DictConfig)->pd.DataFrame:
    df = df.copy()

    df.loc[:, 'assessmentItemID'] = df['assessmentItemID'].str[2:]
    df['assessmentItemID'] = df['assessmentItemID'].astype(int)

    df.loc[:, 'testId'] = df['testId'].str[2:]
    df['testId'] = df['testId'].astype(int)

    return df