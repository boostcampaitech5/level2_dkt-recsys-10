import pandas as pd
import catboost as cb
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

def catboost_classifier(X_train, y_train, X_test, y_test, submission_dataset,config):
    early_stop_rounds = config.catboost_setting.early_stop_rounds
    class_weights = {0: 1.2, 1: 1}

    cat_feature = X_train.select_dtypes(include=['int']).keys().to_list()
    numeric_feature = X_train.select_dtypes(include=['float']).keys().to_list()
    print(f"cat_feature: {cat_feature}")
    print(f"numeric_feature: {numeric_feature}")

    model = cb.CatBoostClassifier(
        **config.catboost_hyperparameters.CatBoostClassifier,
        cat_features= cat_feature,
        # class_weights = class_weights,
    )

    model.fit(X_train, y_train,
                eval_set=(X_test, y_test),
                early_stopping_rounds=early_stop_rounds,
                use_best_model=True)
    
    preds = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, np.where(preds >= 0.5, 1, 0))
    auc = roc_auc_score(y_test, preds)
    print(f"VALID AUC : {auc} ACC : {acc}\n")


    test_pred = model.predict(submission_dataset)
    id_list = range(len(test_pred))
    df = pd.DataFrame({'id': id_list, 'prediction': test_pred})
    df.to_csv(config.dir_name + '/submission.csv', index=False)


    print('---------- sumbit.csv info ----------')
    preds_binary = np.where(test_pred >= 0.5, 1, 0)
    print(f"Count of 0s: {np.sum(preds_binary == 0)}")
    print(f"Count of 1s: {np.sum(preds_binary == 1)}")


if __name__ == "__main__":
    print('test')