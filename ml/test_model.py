from sklearn.metrics import classification_report, accuracy_score
import joblib
from helpers.GetEnv import GetEnv
import argparse
from CustomFactories.SparkSessionFactory import SparkSessionFactory
from pyspark.sql.functions import col
from datetime import date
from GlobalConstants.constants import x_test_schema, y_test_schema
import pandas as pd


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    #parser.add_argument("--mode", required=True, type=str, choices=['test', 'train'], default='train', help="Preprocess data for train / test")
    parser.add_argument("--start_date", required=True, type=str, default='1872-11-30', help="Start date")
    parser.add_argument("--end_date", type=str, required=False, default=str(date.today()), help="End date")

    args = parser.parse_args()

    start_date = args.start_date
    end_date = args.end_date


    _env = GetEnv.get_env_variables()

    spark_session = SparkSessionFactory.create_spark_session()
    df = spark_session.read.format('delta').load(f"{_env['DATA_LAKE_PATH']}/pre_processed_data/featured_result")

    df = df.filter( col('formated_date').between(start_date, end_date) )

    X_test = df.select(x_test_schema)
    Y_test = df.select(y_test_schema)
    

    X_test_pd = X_test.toPandas()
    Y_test_pd = Y_test.toPandas()


    model_path = f"{_env['DATA_LAKE_PATH']}/model/fifa_xgb_model.pkl"

    # Load saved model
    loaded_model = joblib.load(model_path)
    Y_pred = loaded_model.predict(X_test_pd)

    result_df = pd.DataFrame({
        "actual"   : Y_test_pd.squeeze(), #converts 2D dataframe (i.e rows and cols) to 1D dataframe
        "predicted": Y_pred
    })

    print(result_df.head())

    # Predict on test data
    # 

    # print("Actual results   :", list(Y_test_pd))
    # print("Predicted results:", list(Y_pred))