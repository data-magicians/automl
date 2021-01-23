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


if __name__ == "__main__":

    params = json.loads(open("test/params.json", "rb").read())
    spark = None
    mapping = None
    schema = None
    aml_sale = AutoML(spark, mapping, schema, "sale")
    import numpy as np
    # df_p = pd.DataFrame([{"prediction": p, "real": r, "asin": a} for p, r, a in zip(predictions, y, X.reset_index()["asin"])])
    # df_p["prediction_v"] = df_p["prediction"].apply(lambda x: np.exp(x))
    # df_p["real_v"] = df_p["real"].apply(lambda x: np.exp(x))
    # start_time = time.time()
    # aml_sale.preprocess_data()
    # start_time = time.time()
    # aml_sale.train()

    _, _, X, _ = aml_sale.get_data_after_preprocess()
    X = X.groupby(["asin"]).last()
    predictions = aml_sale.predict(X)
    aml_sale.explain(X.reset_index())
    print(1)
