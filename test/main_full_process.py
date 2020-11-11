import sys
import random
sys.path.append("/automl")
from preprocessing.dev_tools import get_cols
from preprocessing.transformers import *
from automl.model_run import *
import pandas as pd
import pickle
import os
import time
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import svm
import json


if __name__ == "__main__":

    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    params = json.loads(open("params.json").read())
    start_time = time.time()

    data_ready = False
    to_train = True
    to_predict = False
    to_explain = False
    path = "df_joined.csv"
    path_predict = ""
    models_names = ["nb", "lr", "knn", "rf", "mlp", "svm", "xgb", "dl", "dl-rnn", "dl-cnn"]
    cols_per = 0.005
    # exclude = ["validity_date_start_month", "validity_date_end_month"]
    exclude = []
    key_cols = ["account_id", "monthtable_rownumber"]
    problems = [{"target": "target_loan", "type": "classification"}, {"target": "target_amount", "type": "regression"}]
    target_cols = [t["target"] for t in problems]
    if to_train:
        if not data_ready:
            df = pd.read_csv(path)
            columns = get_cols(df, key_cols + target_cols + exclude, cols_per)
            print("finish get cols")
            columns["key"] = key_cols
            json.dump(columns, open("columns_type_mapping.json", "w"))
        for i, p in enumerate(problems):
            if not data_ready:
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
                x = features_pipeline(i, X_train, y_train, X_test, y_test, columns, p, key=key_cols[0], date=key_cols[1])
            else:
                x = read_data(p["target"], key_cols)
            if p["type"] == "classification":
                models = [GaussianNB(), LogisticRegression(n_jobs=-1), KNeighborsClassifier(n_jobs=-1),
                          RandomForestClassifier(n_jobs=-1), MLPClassifier(), svm.LinearSVC()]
                          # XGBClassifier(objective="reg:logistic", n_jobs=-1)]
                          # KerasClassifier(build_fn=deeplearning, type="classification", verbose=1),
                          # KerasClassifier(build_fn=deeplearning_rnn, type="classification", verbose=1),
                          # KerasClassifier(build_fn=deeplearning_cnn, type="classification", verbose=1)]
            else:
                models = [None, ElasticNet(), KNeighborsRegressor(n_jobs=-1),
                          RandomForestRegressor(n_jobs=-1), MLPRegressor(), svm.LinearSVR()]
                          # XGBRegressor(objective="reg:linear", n_jobs=-1)]
                          # KerasRegressor(build_fn=deeplearning, type="regression", verbose=1),
                          # KerasRegressor(build_fn=deeplearning_rnn, type="regression", verbose=1),
                          # KerasRegressor(build_fn=deeplearning_cnn, type="regression", verbose=1)]
            for index, model in enumerate(models):
                models_2_run = [(x[1], model, params["hyperparameters"][p["type"]][models_names[index]], x[2], x[3], x[4], x[5],
                                 models_names[index], x[6], p["type"])]
                results = [model_pipeline_run_unpack(x) for x in models_2_run]
                folder = p["target"]
                path = os.path.dirname(os.path.realpath(__file__))
                if len(results) > 0 and results[0] is not None:
                    if not os.path.exists(path + "/results"):
                        os.mkdir(path + "/results")
                    for row in results:
                        try:
                            if "dl" not in models_names[index]:
                                file = "/results/model_results_{}_{}.p".format(folder, models_names[index])
                                pickle.dump(row["grid"], open(path + file, "wb"))
                            else:
                                file = "/results/model_result_{}_{}.yaml".format(folder, models_names[index])
                                grid = row["grid"].best_estimator_.steps[0][1].model.save(path + file)
                                model_b = grid.best_estimator_.model
                                model_yaml = model_b.to_yaml()
                                with open(file, "w") as yaml_file:
                                    yaml_file.write(model_yaml)
                                # serialize weights to HDF5
                                file = "/results/model_result_{}_{}.h5".format(folder, models_names[index])
                                model_b.save_weights(file)
                            del row["grid"]
                            del grid
                        except Exception as e:
                            print(e)
                    file = "/results/model_results_{}.csv".format(folder)
                    file_json = "/results/model_results_{}_{}.json".format(folder, models_names[index])
                    if not os.path.exists(file):
                        d = pd.DataFrame([r for r in results if r is not None])
                        d.to_csv(path + file, index=False)
                        js = pd.read_csv(path + file)
                        js["report"] = js["report"].apply(string_2json)
                        js = js.to_dict(orient="records")[0]
                        json.dump(js, open("results/model_results_{}_{}.json".format(folder, models_names[index]), "w"))
                    else:
                        d = pd.DataFrame([r for r in results if r is not None])
                        d.to_csv(path + file, index=False).to_csv(path + file, index=False, mode="a", header=False)
                        js = pd.read_csv(path + file)
                        js["report"] = js["report"].apply(string_2json).to_dict(orient="records")[0]
                        js = js.to_dict(orient="records")[0]
                        json.dump(js, open("results/model_results_{}_{}.json".format(folder, models_names[index]), "w"))
    for i, p in enumerate(problems):
        x = read_data(p["target"], key_cols)
        folder = p["target"]
        for models_name in models_names:
            if "dl" not in models_name:
                file = "/results/model_results_{}_{}.p".format(folder, models_name)
                model = pickle.load(open(path + file, "rb"))
            else:
                file = "/results/{}/".format(p["target"])
                model = load_model(file)
        X_train = pd.read_csv("preprocess_results/X_train_{}.csv".format(p["target"])).set_index(key_cols)
        X_test = pd.read_csv("preprocess_results/X_train_{}.csv".format(p["target"])).set_index(key_cols)
        data = pd.concat([X_train, X_test])
        if to_predict:
            result = pd.DataFrame(model.predict(data))
            result.to_csv(path_predict)
        if to_explain:
            explain(model, data, models_names[index], p["target"], key_cols)
    print("the total time in minutes is: {}".format((time.time() - start_time) / 60))
