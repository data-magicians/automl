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
    aml = AutoML(spark, mapping, schema, "target_loan")
    # aml.create_panel()
    # aml.preprocess_data()
    # aml.train()
    _, _, X, _ = aml.get_data_after_preprocess()
    X = X.groupby(["account_id"]).last()
    predictions = aml.predict(X)
    aml.explain(X)
