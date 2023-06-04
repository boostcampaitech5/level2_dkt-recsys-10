import pandas as pd
import hydra
import omegaconf

def dataloader_decorator(func):
    def wrapper(config: omegaconf.DictConfig):
        print('----------- dataloader -----------')
        dataset_path = hydra.utils.to_absolute_path(config.dataset.dataset)
        submission_path = hydra.utils.to_absolute_path(config.dataset.submission)
        keys = config.dataset_feature.feature
        keys.append(config.dataset_feature.label)

        dataset_csv = pd.read_csv(dataset_path)
        submission_csv = pd.read_csv(submission_path)

        dataset_data = dataset_csv[keys]
        submission_data = submission_csv[keys]

        print('---------- load success ----------')
        return func(dataset_data, submission_data, config)
    return wrapper


if __name__ == "__main__":
    print('data_loader')