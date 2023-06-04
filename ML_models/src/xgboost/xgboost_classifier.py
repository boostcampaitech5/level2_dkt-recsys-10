import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder

def xgboost_classifier(X_train, y_train, X_test, y_test, submission_dataframe, config):
    # X_train에서 int 열 인덱스 가져오기
    int_columns = X_train.select_dtypes(include=['int']).columns
    # int 열을 category로 변경
    X_train[int_columns] = X_train[int_columns].astype('category')

    # X_test에서 int 열 인덱스 가져오기
    int_columns = X_test.select_dtypes(include=['int']).columns
    # int 열을 category로 변경
    X_test[int_columns] = X_test[int_columns].astype('category')

    # 범주형 데이터 열 인덱스 가져오기
    cat_columns = X_train.select_dtypes(include=['category']).columns
    print(f'cat_columns = {cat_columns}')
    # OrdinalEncoder 객체 생성
    encoder = OrdinalEncoder()

    # X_train과 X_test의 범주형 데이터를 Ordinal 인코딩으로 변환
    X_train[cat_columns] = encoder.fit_transform(X_train[cat_columns])
    X_test[cat_columns] = encoder.transform(X_test[cat_columns])

    # XGBoost Classifier 모델 설정
    model = xgb.XGBClassifier(
        **config.catboost_hyperparameters.XGBClassifier
    )

    # 모델 학습
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)])

    # 예측
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # 정확도 계산
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    # AUC 계산
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"AUC: {auc}")