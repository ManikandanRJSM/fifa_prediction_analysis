import pandas as pd
from delta.tables import DeltaTable
from helpers.GetEnv import GetEnv
from GlobalConstants.constants import x_training_schema, y_training_schema
from CustomFactories.SparkSessionFactory import SparkSessionFactory
from pathlib import Path
from pyspark.sql.functions import col
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score

import xgboost as xgb

import joblib
import argparse


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    #parser.add_argument("--mode", required=True, type=str, choices=['test', 'train'], default='train', help="Preprocess data for train / test")
    parser.add_argument("--start_date", required=False, type=str, default='1872-11-30', help="Start date")
    parser.add_argument("--end_date", required=False, type=str, default='2024-12-31', help="End date")

    args = parser.parse_args()

    start_date = args.start_date
    end_date = args.end_date
    
    _env = GetEnv.get_env_variables()

    model_path = f"{_env['DATA_LAKE_PATH']}/model/fifa_xgb_model.pkl"
    spark_session = SparkSessionFactory.create_spark_session()

    df = spark_session.read.format('delta').load(f"{_env['DATA_LAKE_PATH']}/featured_result/vectors")

    df = df.filter( col('formated_date').between(start_date, end_date) )

    pd = df.toPandas()
    
    X_train = pd[x_training_schema]
    y_train = pd[y_training_schema]

    sample_weights = compute_sample_weight('balanced', y_train)
    model = xgb.XGBClassifier(
        n_estimators     = 500,
        max_depth        = 7,     # increase from 6
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        min_child_weight = 3,     # reduce from 4
        gamma            = 0.01,  # reduce from 0.05
        reg_lambda       = 0.5,   # reduce from 1.0
        reg_alpha        = 0.05,  # reduce from 0.1
        objective        = 'binary:logistic',
        random_state     = 42
    )

    model.fit(
        X_train, y_train,
        sample_weight=sample_weights
    )

    joblib.dump(model, model_path)

    train_accuracy = accuracy_score(y_train, model.predict(X_train))

    print(f"Train Accuracy : {round(train_accuracy * 100, 2)}%")

    print("Model saved!")
    spark_session.stop()