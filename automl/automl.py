# -*- encoding: utf-8 -*-
from automl.preprocessing.preprocess_utils import *
from automl.modeling.modeling import *
import logging
import sys
import random
from automl.dev_tools import *

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
        self._X_return = None


    def create_panel(self):
        """
        create data panel
        """
        params = json.loads(open("params.json", "rb").read())
        get_data_params = params["raw_data"]
        data = get_data()
        logging.info("Data tables loaded: " + " ".join(list(data.keys())))
        dates_df = create_dates_table(get_data_params["start_date"], get_data_params["end_date"])
        logging.info("Dates table loaded with " + get_data_params["start_date"] + " to " + get_data_params["end_date"] + " and in shape of " + str(dates_df.shape))
        trans_df = feature_engineering_transactions(data["trans_train"], dates_df)
        logging.info("Trans table feature engineering complete with shape of" + str(trans_df.shape))
        df = clean_and_join_other_data(data)
        logging.info("Main df table feature engineering complete with shape of" + str(df.shape))
        df_joined = join_data_final(df, trans_df, data)
        logging.info("Final joined df complete with shape of" + str(df_joined.shape))
        spark_df_joined = self.session.createDataFrame(df_joined)
        spark_df_joined.createOrReplaceTempView(params["raw_data"]["raw_data_name"])


    def get_raw_data(self):

        return self.session.sql("select * from spark_df_joined")


    def preprocess_data(self):

        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        params = json.loads(open("params.json").read())
        cols_per = params["cols_per"]
        exclude_cols = params["exclude_cols"]
        key_cols = params["key_cols"]
        problems = params["problems"]
        target_cols = [t["target"] for t in problems]
        df = self.session.sql("select * from spark_df_joined").toPandas()
        columns = get_cols(df, key_cols + target_cols + exclude_cols, cols_per)
        logging.info("finish get cols")
        columns["key"] = key_cols
        json.dump(columns, open("columns_type_mapping.json", "w"))
        X_return = []
        for i, p in enumerate(problems):
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
            x = features_pipeline(i, X_train, y_train, X_test, y_test, columns, p, self.session, key=key_cols[0], date=key_cols[1])
            X_return.append(x)
        self._X_return = X_return
        return X_return


    def get_data_after_preprocess(self):

        return self._X_return


    def train(self):

        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        params = json.loads(open("params.json").read())
        problems = params["problems"]
        models_names = params["models_names"]

        for i, p in enumerate(problems):
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
            x = self._X_return[i]
            for index, model in enumerate(models):
                models_2_run = [(x[1], model, params["hyperparameters"][p["type"]][models_names[index]], x[2], x[3], x[4], x[5], models_names[index], x[6], p["type"])]
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
                                save(open(path + file, "wb"), row["grid"])
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
                    if not os.path.exists(file):
                        d = pd.DataFrame([r for r in results if r is not None])
                        d.to_csv(path + file, index=False)
                        js = pd.read_csv(path + file)
                        os.remove(path + file)
                        js["report"] = js["report"].apply(string_2json)
                        js = js.to_dict(orient="records")[0]
                        save(js, open("results/model_results_{}_{}.json".format(folder, models_names[index]), "w"))
                    else:
                        d = pd.DataFrame([r for r in results if r is not None])
                        d.to_csv(path + file, index=False).to_csv(path + file, index=False, mode="a", header=False)
                        js = pd.read_csv(path + file)
                        os.remove(path + file)
                        js["report"] = js["report"].apply(string_2json).to_dict(orient="records")[0]
                        js = js.to_dict(orient="records")[0]
                        save(js, open("results/model_results_{}_{}.json".format(folder, models_names[index]), "w"))

    def get_best_model(self):

        pass

    def evaluate(self):

        pass

    def get_metrics(self):

        pass

    def explain(self):

        pass
