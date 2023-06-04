import hydra
import omegaconf
from hydra import utils
import os
import shutil

from src.data_process import data_process
from src.train_test_split import train_test_split
from src.catboost.catboost_classifier import catboost_classifier
from src.catboost.catboost_regressor import catboost_regressor
from src.xgboost.xgboost_classifier import xgboost_classifier


@hydra.main(version_base=None, config_path='.', config_name="config_hydra")
def train(config: omegaconf.DictConfig) -> None:
    omegaconf.OmegaConf.set_struct(config, False)
    
    dataset__dataframe, submission_dataframe = data_process(config)
    X_train, y_train, X_test, y_test, submission_dataframe = train_test_split(dataset__dataframe, submission_dataframe, config)
    
    # catboost logging
    catboost_log_file = config.catboost_log_file
    destination_file = config.destination_file

    print(y_train.value_counts())
    print(y_test.value_counts())
    if config.model.select_model == 'catboost_classifier':
        catboost_classifier(X_train, y_train, X_test, y_test, submission_dataframe, config)
        shutil.copyfile(catboost_log_file, destination_file)
    elif config.model.select_model == 'catboost_regressor':
        catboost_regressor(X_train, y_train, X_test, y_test, submission_dataframe, config)
        shutil.copyfile(catboost_log_file, destination_file)
    elif config.model.select_model == 'xgboost_classifier':
        xgboost_classifier(X_train, y_train, X_test, y_test, submission_dataframe, config)
    
    
    
if __name__ == "__main__":
    train()
