"""
11.11.2020
test input from datomize
"""
from pyspark.sql.session import SparkSession
from automl.automl import AutoML
from google.protobuf.json_format import MessageToDict
from pyspark import SparkConf
# from automl.test import simple_pb2
import os
# del os.environ['PYSPARK_SUBMIT_ARGS']
# os.environ["JAVA_HOME"] = r'C:\Program Files\Java\jre1.8.0_271'
if __name__ == "__main__":

    sc_conf = SparkConf()
    sc_conf.setAppName("etl")
    # sc_conf.set('spark.executor.memory', '16g')
    # sc_conf.set('spark.driver.memory', '16g')
    # sc_conf.set('spark.debug.maxToStringFields', 500)
    # spark = SparkSession.builder.config(conf=sc_conf).getOrCreate()
    # spark = SparkSession.builder.getOrCreate()
    # dict_obj = MessageToDict(simple_pb2)
    spark = None
    mapping = None
    schema = None
    aml = AutoML(spark, mapping, schema)
    # aml.create_panel()
    aml.preprocess_data()
    aml.train()
    _, _, X, _ = aml.get_data_after_preprocess({"target": "target_loan"})
    X = X.groupby(["account_id"]).last().reset_index()
    # predictions = aml.predict(X)
    # aml.explain(X)