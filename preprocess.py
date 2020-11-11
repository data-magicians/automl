import pandas as pd
import numpy as np
import os
from datetime import datetime
import pandasql as ps
from datetime import timedelta
import logging
import sys
import preprocess_utils

logging.basicConfig(format='%(asctime)s     %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    stream=sys.stdout,
                    level='INFO')


data = preprocess_utils.get_data()
logging.info("Data tables loaded: " + " ".join(list(data.keys())))
start_date = "1993-01-01"
end_date = "1997-12-31"
dates_df = preprocess_utils.create_dates_table(start_date, end_date)
logging.info("Dates table loaded with "+start_date + " to " + end_date + " and in shape of "+ str(dates_df.shape))
trans_df = preprocess_utils.feature_engineering_transactions(data["trans_train"], dates_df)
logging.info("Trans table feature engineering complete with shape of" + str(trans_df.shape))
df = preprocess_utils.clean_and_join_other_data(data)
logging.info("Main df table feature engineering complete with shape of" + str(df.shape))
df_joined = preprocess_utils.join_data_final(df,trans_df,data)
logging.info("Final joined df complete with shape of" + str(df_joined.shape))

# Pandas to Spark
# spark_df_joined = spark_session.createDataFrame(df_joined)
# spark_df_joined.createOrReplaceTempView("spark_df_joined")