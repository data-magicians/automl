# -*- encoding: utf-8 -*-
from automl.preprocessing import preprocess_utils
import logging
import sys
import json


class AutoML:

    logging.basicConfig(format='%(asctime)s     %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        stream=sys.stdout,
                        level='INFO')

    def __init__(self, session, schema, mapping):

        self.session = session
        self.schema = schema
        self.mapping = mapping


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

        pass

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
