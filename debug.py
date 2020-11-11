import sys
import random
sys.path.append("/automl")
from sklearn import metrics
from preprocessing.dev_tools import get_cols
from preprocessing.pipelines import MLPipeline
from preprocessing.transformers import *
from model_run import *
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import shap
import pickle
import os
import time
from xgboost import XGBClassifier, XGBRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import svm
from keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
import json


if __name__ == "__main__":

    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    params = json.loads(open("params.json").read())
    start_time = time.time()
    to_train = True
    data_ready = True
    path = "df_joined.csv"
    models_names = ["nb", "lr", "knn", "rf", "mlp", "svm", "xgb", "dl", "dl-rnn", "dl-cnn"]
    cols_per = 0.005
    # exclude = ["validity_date_start_month", "validity_date_end_month"]
    exclude = []
    key_cols = ["account_id", "monthtable_rownumber"]
    problems = [{"target": "target_loan", "type": "classification"}, {"target": "target_amount", "type": "regression"}]
    target_cols = [t["target"] for t in problems]

    explain(None, None, "rf", "target_loan", key_cols)
    # target = "target_loan"
    # for model in ["nb", "lr", "knn", "mlp", "svm", "rf"]:
    #     js = json.load(open("results/model_results_{}_{}.json".format(target, model), "rb"))
    #     js["report"] = string_2json(js["report"])
    #     json.dump(js, open("results/model_results_{}_{}.json".format(target, model), "w"))

    print("the total time in minutes is: {}".format((time.time() - start_time) / 60))
