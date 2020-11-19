# -*- encoding: utf-8 -*-
from automl.preprocessing import preprocess_utils
from automl.preprocessing.dev_tools import *
import logging
import sys
import json
import os
import time
import random


class AutoML:

    logging.basicConfig(format='%(asctime)s     %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        stream=sys.stdout,
                        level='INFO')

    def __init__(self, session, schema, mapping):

        self.session = session
        self.schema = schema
        self.mapping = mapping
        self._data_ready = True


    def create_panel(self):
        """
        create data panel
        """
        params = json.loads(open("params.json", "rb").read())
        get_data_params = params["raw_data"]
        data = preprocess_utils.get_data()
        logging.info("Data tables loaded: " + " ".join(list(data.keys())))
        dates_df = preprocess_utils.create_dates_table(get_data_params["start_date"], get_data_params["end_date"])
        logging.info("Dates table loaded with " + get_data_params["start_date"] + " to " + get_data_params["end_date"] + " and in shape of " + str(dates_df.shape))
        trans_df = preprocess_utils.feature_engineering_transactions(data["trans_train"], dates_df)
        logging.info("Trans table feature engineering complete with shape of" + str(trans_df.shape))
        df = preprocess_utils.clean_and_join_other_data(data)
        logging.info("Main df table feature engineering complete with shape of" + str(df.shape))
        df_joined = preprocess_utils.join_data_final(df, trans_df, data)
        logging.info("Final joined df complete with shape of" + str(df_joined.shape))
        spark_df_joined = self.session.createDataFrame(df_joined)
        spark_df_joined.createOrReplaceTempView(params["raw_data"]["raw_data_name"])

    def get_raw_data(self):

        return self.session.sql("select * from spark_df_joined")

    def preprocess_data(self):

        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        params = json.loads(open("params.json").read())
        start_time = time.time()
        cols_per = params["cols_per"]
        exclude_cols = params["exclude_cols"]
        key_cols = params["key_cols"]
        problems = params["problems"]
        target_cols = [t["target"] for t in problems]
        df = self.session.sql("select * from spark_df_joined").toPandas()
        columns = get_cols(df, key_cols + target_cols + exclude_cols, cols_per)
        print("finish get cols")
        columns["key"] = key_cols
        json.dump(columns, open("columns_type_mapping.json", "w"))
        for i, p in enumerate(problems):
            if not self._data_ready:
                columns["target"] = [p["target"]]
                accounts_train, accounts_test = [], []
                for account in df[key_cols[0]].unique():
                    rand = random.random()
                    if rand < params["train_test_per"]:
                        accounts_train.append(account)
                    else:
                        accounts_test.append(account)
                df_train = df[df[key_cols[0]].isin(accounts_train)].set_index(key_cols)[
                    columns["numeric"] + columns["categoric"] + columns["target"]]
                df_test = df[df[key_cols[0]].isin(accounts_test)].set_index(key_cols)[
                    columns["numeric"] + columns["categoric"] + columns["target"]]
                X_train = df_train.drop(p["target"], axis=1)
                y_train = df_train[p["target"]]
                X_test = df_test.drop(p["target"], axis=1)
                y_test = df_test[p["target"]]
                x = features_pipeline(i, X_train, y_train, X_test, y_test, columns, self.spark, p, key=key_cols[0],
                                      date=key_cols[1])
                self._data_ready = True
            else:
                x = read_data(p["target"], key_cols)
        return ()

    def get_data_after_preprocess(self):

        pass

    def train(self):

        pass

    def get_best_model(self):

        pass

    def evaluate(self):

        pass

    def get_metrics(self):

        pass

    def explain(self):

        pass
