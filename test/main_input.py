"""
11.11.2020
test input from datomize
"""
from pyspark.sql.session import SparkSession
from automl.automl import AutoML
from google.protobuf.json_format import MessageToDict
# from automl.test import simple_pb2


if __name__ == "__main__":

    spark = SparkSession.builder.getOrCreate()
    # dict_obj = MessageToDict(simple_pb2)
    mapping = None
    schema = None
    aml = AutoML(spark, mapping, schema)
    aml.create_panel()
    aml.preprocess_data()
    aml.train()
    aml.evaluate()
    aml.explain()
