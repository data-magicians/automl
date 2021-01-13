"""
11.11.2020
test input from datomize
"""
from pyspark.sql.session import SparkSession
from automl.automl import AutoML
from pyspark import SparkConf
from google.protobuf.json_format import MessageToDict
# from automl.test import simple_pb2
# import os


if __name__ == "__main__":

    sc_conf = SparkConf()
    sc_conf.setAppName("etl")
    sc_conf.set('spark.executor.memory', '16g')
    sc_conf.set('spark.driver.memory', '16g')
    sc_conf.set('spark.debug.maxToStringFields', 500)
    spark = SparkSession.builder.config(conf=sc_conf).getOrCreate()
    # dict_obj = MessageToDict(simple_pb2)
    mapping = None
    schema = None
    aml_loan = AutoML(spark, mapping, schema, "target_loan")
    aml_amount = AutoML(spark, mapping, schema, "target_amount")
    aml_churn = AutoML(spark, mapping, schema, "target_churn")
    aml_amount = AutoML(spark, mapping, schema, "gender")
    # aml_loan.preprocess_data()
    aml_loan.train()
    aml_loan.preprocess_data()
    aml_loan.train()
    aml_loan.preprocess_data()
    aml_loan.train()
    aml_loan.preprocess_data()
    aml_loan.train()
    _, _, X, _ = aml_loan.get_data_after_preprocess()
    X = X.groupby(["account_id"]).last()
    predictions = aml_loan.predict(X)
    aml_loan.explain(X.reset_index())
    _, _, X, _ = aml_amount.get_data_after_preprocess()
    X = X.groupby(["account_id"]).last()
    predictions = aml_amount.predict(X)
    aml_amount.explain(X.reset_index())
    _, _, X, _ = aml_churn.get_data_after_preprocess()
    X = X.groupby(["account_id"]).last()
    predictions = aml_churn.predict(X)
    aml_churn.explain(X.reset_index())
