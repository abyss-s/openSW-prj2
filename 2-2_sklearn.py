# Project #2-2 Data analysis with sklearn
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer


# 1. 데이터를 해당 시즌(연도) 기준으로 정렬
def sort_dataset(dataset_df):
    sorted_df = dataset_df.sort_values(by='year')
    return sorted_df


# 2. 데이터 스플릿
def split_dataset(dataset_df):
    dataset_df['salary'] *= 0.001
    train_df = dataset_df.iloc[:1718]
    test_df = dataset_df.iloc[1718:]
    X_train = extract_numerical_cols(train_df)
    X_test = extract_numerical_cols(test_df)
    Y_train = train_df['salary']
    Y_test = test_df['salary']
    return X_train, X_test, Y_train, Y_test


# 3. 숫자 특성만 추출
def extract_numerical_cols(dataset_df):
    cols_list = ['age', 'G', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'HBP', 'SO', 'GDP',
                 'fly', 'war']
    numerical_cols = dataset_df[cols_list]
    return numerical_cols


# 4. 의사 결정 트리 모델 훈련
def train_predict_decision_tree(X_train, Y_train, X_test):
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, Y_train)
    dt_predictions = model.predict(X_test)
    return dt_predictions


# 5. 랜덤 포레스트 모델을 훈련
def train_predict_random_forest(X_train, Y_train, X_test):
    n = SimpleImputer(strategy='mean')
    X_train_rd = pd.DataFrame(n.fit_transform(X_train), columns=X_train.columns)
    X_test_rd = pd.DataFrame(n.transform(X_test), columns=X_test.columns)
    m = RandomForestRegressor(random_state=42)
    m.fit(X_train_rd, Y_train)
    rf_predictions = m.predict(X_test_rd)
    return rf_predictions


# 6. SVM 훈련 및 예측
def train_predict_svm(X_train, Y_train, X_test):
    n = Pipeline([('scaler', StandardScaler()), ('svm', SVR())])
    n.fit(X_train, Y_train)
    svm_predictions = n.predict(X_test)
    return svm_predictions


# 7. RMSE 계산
def calculate_RMSE(labels, predictions):
    RMSE_predictions = mean_squared_error(labels, predictions, squared=False)
    return RMSE_predictions


if __name__ == '__main__':
    file_path = '2019_kbo_for_kaggle_v2.csv'
    # DO NOT MODIFY THIS FUNCTION UNLESS PATH TO THE CSV MUST BE CHANGED.
    data_df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')

    sorted_df = sort_dataset(data_df)
    X_train, X_test, Y_train, Y_test = split_dataset(sorted_df)

    X_train = extract_numerical_cols(X_train)
    X_test = extract_numerical_cols(X_test)

    dt_predictions = train_predict_decision_tree(X_train, Y_train, X_test)
    rf_predictions = train_predict_random_forest(X_train, Y_train, X_test)
    svm_predictions = train_predict_svm(X_train, Y_train, X_test)

    print("Decision Tree Test RMSE: ", calculate_RMSE(Y_test, dt_predictions))
    print("Random Forest Test RMSE: ", calculate_RMSE(Y_test, rf_predictions))
    print("SVM Test RMSE: ", calculate_RMSE(Y_test, svm_predictions))
