"""
11.11.2020
test input from datomize
"""
from pyspark.sql.session import SparkSession
from automl.automl import AutoML
from automl.dev_tools import send_email
import time
import json
from pyspark import SparkConf
from google.protobuf.json_format import MessageToDict
# from automl.test import simple_pb2
# import os


if __name__ == "__main__":

    # sc_conf = SparkConf()
    # sc_conf.setAppName("etl")
    # sc_conf.set('spark.executor.memory', '16g')
    # sc_conf.set('spark.driver.memory', '16g')
    # sc_conf.set('spark.debug.maxToStringFields', 500)
    # spark = SparkSession.builder.config(conf=sc_conf).getOrCreate()
    # dict_obj = MessageToDict(simple_pb2)
    params = json.loads(open("test/params.json", "rb").read())
    start_time_total = time.time()
    try:
        spark = None
        mapping = None
        schema = None
        aml_loan = AutoML(spark, mapping, schema, "target_loan")
        aml_amount = AutoML(spark, mapping, schema, "target_amount")
        aml_churn = AutoML(spark, mapping, schema, "target_churn")
        aml_gender = AutoML(spark, mapping, schema, "gender")

        # start_time = time.time()
        # aml_loan.preprocess_data()
        # send_email(params["email_notifications"], "the loan preprocess finished in: {} minutes".format((time.time() - start_time) / 60), "datomize")
        # start_time = time.time()
        # aml_loan.train()
        # send_email(params["email_notifications"], "the loan train finished in: {} minutes".format((time.time() - start_time) / 60), "datomize")

        start_time = time.time()
        aml_amount.preprocess_data()
        send_email(params["email_notifications"], "the amount preprocess finished in: {} minutes".format((time.time() - start_time) / 60), "datomize")
        start_time = time.time()
        aml_amount.train()
        send_email(params["email_notifications"], "the amount train finished in: {} minutes".format((time.time() - start_time) / 60), "datomize")

        start_time = time.time()
        aml_churn.preprocess_data()
        send_email(params["email_notifications"], "the churn preprocess finished in: {} minutes".format((time.time() - start_time) / 60), "datomize")
        start_time = time.time()
        aml_churn.train()
        send_email(params["email_notifications"], "the churn train finished in: {} minutes".format((time.time() - start_time) / 60), "datomize")

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

        send_email(params["email_notifications"], "the total finished in: {} minutes".format((time.time() - start_time_total) / 60), "datomize")
        print("the total finished in: {} minutes".format((time.time() - start_time_total) / 60))
    except Exception as e:
        print(e)
        send_email(params["email_notifications"], "error: {},  finished in: {} minutes".format(str(e), (time.time() - start_time_total) / 60), "datomize")
