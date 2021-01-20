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

    def __init__(self, session, schema, mapping, problem):

        self.session = session
        self.schema = schema
        self.mapping = mapping
        self._data_ready = True
        self.problem = problem
        self._X_return = None

    def create_panel(self, test=True):
        """
        create data panel
        """
        params = json.loads(open("automl/params.json", "rb").read())
        get_data_params = params["raw_data"]
        if not test:
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
        else:
            spark_df_joined = self.session.read.option("header", "true").csv("test/df_joined.csv")
            spark_df_joined.createOrReplaceTempView(params["raw_data"]["raw_data_name"])

    def get_raw_data(self):

        return self.session.sql("select * from spark_df_joined")

    def preprocess_data(self, only_one_return=True):

        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        params = json.loads(open("params.json").read())
        cols_per = params["cols_per"]
        key_cols = params["key_cols"]
        problems = [x for x in params["problems"] if self.problem in x["target"]]
        exclude_cols = [x["exclude_cols"] for x in params["problems"] if self.problem in x["target"]][0]
        target_cols = [t["target"] for t in problems]
        top_n = params["feature_selection"]
        # todo remove the pandas and uncomment the spark to pandas
        # df = self.session.sql("select * from spark_df_joined").toPandas()
        # df = pd.read_csv('C:\\Users\\Administrator\\PycharmProjects\\automl\\test\\df_joined.csv').head(1000)
        df = pd.read_csv('C:\\Users\\Administrator\\PycharmProjects\\automl\\test\\df_joined.csv')
        try:
            df = df.drop(exclude_cols, axis=1)
        except Exception as e:
            print("can't remove the excluded features")
        columns = get_cols(df, key_cols + target_cols, cols_per)
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
            x = features_pipeline(i, X_train, y_train, X_test, y_test, columns, p, self.session, key=key_cols[0], date=key_cols[1], top_n=top_n)
            X_return.append(x)
            if only_one_return:
                return X_return
        self._X_return = X_return
        return X_return

    def get_data_after_preprocess(self, spark=False):
        """

        """
        path = os.path.dirname(__file__)
        params = json.loads(open(path + "/params.json").read())
        key_cols = params["key_cols"]
        row = [x for x in params["problems"] if self.problem in x["target"]][0]

        if spark:
            X_train = self.session.sql("select * from preprocess_results.X_train_{}".format(row["target"]))
            y_train = self.session.sql("select * from preprocess_results.y_train_{}.csv".format(row["target"]))
            X_test = self.session.sql("select * from preprocess_results.X_test_{}.csv".format(row["target"]))
            y_test = self.session.sql("select * from preprocess_results.y_test_{}.csv".format(row["target"]))
        else:
            X_train = pd.read_csv(path + "/preprocess_results/X_train_{}.csv".format(row["target"]))
            y_train = pd.read_csv(path + "/preprocess_results/y_train_{}.csv".format(row["target"])).values.flatten()
            X_test = pd.read_csv(path + "/preprocess_results/X_test_{}.csv".format(row["target"]))
            y_test = pd.read_csv(path + "/preprocess_results/y_test_{}.csv".format(row["target"])).values.flatten()

        X_train = X_train.set_index(key_cols)
        X_test = X_test.set_index(key_cols)

        return X_train, y_train, X_test, y_test

    def train(self):
        """

        """
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        params = json.loads(open("params.json").read())
        problems = [x for x in params["problems"] if self.problem in x["target"]]
        models_names = params["models_names"]

        for i, p in enumerate(problems):
            best_score = None
            best_model = None
            js_best = None
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
            x = self.get_preprocess_pipeline()
            X_train, y_train, X_test, y_test = self.get_data_after_preprocess()
            x = (x[0], x[1], X_train, y_train, X_test, y_test, x[2], x[3])
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
                            file = "/results/model_results_{}_{}".format(folder, models_names[index])
                            save(open(path + file, "wb"), row["grid"])
                            del row["grid"]
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
                        js["columns"] = js["columns"].split("*|*")
                        js["hyperparameters"] = eval(js["hyperparameters"])
                        file = "results/model_results_{}_{}.json".format(folder, models_names[index])
                        json.dump(js, open(file, "w"))
                    else:
                        d = pd.DataFrame([r for r in results if r is not None])
                        d.to_csv(path + file, index=False).to_csv(path + file, index=False, mode="a", header=False)
                        js = pd.read_csv(path + file)
                        os.remove(path + file)
                        js["report"] = js["report"].apply(string_2json)
                        js = js.to_dict(orient="records")[0]
                        js["columns"] = js["columns"].split("*|*")
                        js["hyperparameters"] = eval(js["hyperparameters"])
                        file = "results/model_results_{}_{}.json".format(folder, models_names[index])
                        json.dump(js, open(file, "w"))
                    if p["type"] == "regression":
                        if best_score is None or best_score < js["report"]["test_r2_score"]:
                            best_model = "{}/{}".format(path, file)
                            best_score = js["report"]["test_r2_score"]
                            js_best = js
                    else:
                        if best_score is None or best_score < js["report"]["macro avg_test"]["f1-score"]:
                            best_model = "{}/{}".format(path, file)
                            best_score = js["report"]["macro avg_test"]["f1-score"]
                            js_best = js
            # save(open(best_model + "_best_model", "w"), js)
            json.dump(js_best, open(best_model.split(".")[0] + "_best_model" + ".json", "w"))

    def get_best_model(self, extra_data=False, explain=False):

        path = os.path.dirname(os.path.realpath(__file__)) + "/results/"
        files = os.listdir(path)
        if explain:
            files_best = [f.replace("_best_model.json", "") for f in files if "_rf" in f and self.problem in f and "json" not in f]
        else:
            files_best = [f.replace("_best_model.json", "") for f in files if "best_model" in f and self.problem in f]
        model = None
        metrics = None
        for f in files_best:
            # a = pickle.load(load(open(path + f, "rb")))
            model = load(open(path + f, "rb"))
            metrics = json.loads(open(path + f + ".json").read())
        if not extra_data:
            return model
        else:
            return model, files_best[0], metrics

    def get_preprocess_pipeline(self):

        path = os.path.dirname(os.path.realpath(__file__))
        params = json.loads(open(path + "/params.json").read())
        problem = [x for x in params["problems"] if self.problem in x["target"]][0]
        pipeline = None
        file_name = path + "/preprocess_results/preprocess_pipeline_{}".format(problem["target"])
        try:
            pipeline = load(file_name)
        except Exception as e:
            print("problem in get_preprocess_pipeline")
            print(e)
        self._X_return = pipeline
        return pipeline

    def predict(self, X):

        model = self.get_best_model()
        return model.predict(X)

    @staticmethod
    def shap_analysis(df, shap_values, features, id_field_name="", entitis=[], top_n=200, number_of_features_use=10):
        # create a small df with only x amount of customers for shap explanation

        # we need to look at absolute values
        shap_feature_df_abs = pd.DataFrame(shap_values).abs()
        # Apply Decorate-Sort row-wise to our df, and slice the top-n columns within each row...
        sort_decr2_topn = lambda row, nlargest=top_n: sorted(pd.Series(zip(shap_feature_df_abs.columns, row)),
                                                           key=lambda cv: -cv[1])[:nlargest]
        tmp = shap_feature_df_abs.apply(sort_decr2_topn, axis=1)
        # then your result (as a pandas DataFrame) is...
        np.array(tmp)
        list_of_user_dfs = []

        for i in range(0, len(entitis)):
            one_user_id = entitis[i]
            weights_one_user = tmp[i]
            # add weight
            user_weight_df = pd.DataFrame(weights_one_user)
            user_weight_df.columns = ["feature", "weight_val"]
            user_weight_df = user_weight_df.sort_values(by="feature", ascending=False)
            user_weight_df['sum'] = user_weight_df['weight_val'].sum()
            user_weight_df['weight'] = user_weight_df['weight_val'] / user_weight_df['sum']
            user_weight_df['weight'] = user_weight_df['weight'].round(2)
            del user_weight_df['sum']
            del user_weight_df['weight_val']
            user_weight_df = user_weight_df.sort_values(by="weight", ascending=False)
            user_weight_df["feature"] = df.columns[user_weight_df["feature"].tolist()]
            # join original value
            user_original_values = df[df[id_field_name].isin([one_user_id])]
            user_original_values = user_original_values[features].T.reset_index()
            user_original_values.columns = ["feature", "feature_value"]
            user_weight_df = pd.merge(user_weight_df, user_original_values, on="feature")
            # rank the feature weight
            user_weight_df["feature_rank"] = user_weight_df.index + 1
            # add user id
            user_weight_df[id_field_name] = one_user_id
            user_weight_df = user_weight_df[user_weight_df["feature_rank"] <= number_of_features_use]
            list_of_user_dfs.append(user_weight_df)
        final_shap_df_1 = pd.concat(list_of_user_dfs)
        return final_shap_df_1

    def get_metrics_of_models(self):

        path = os.path.dirname(os.path.realpath(__file__)) + "/results/"
        files = [f for f in os.listdir(path) if self.problem in f and ".json" in f]
        metrics = []
        for f in files:
            try:
                metric = json.loads(open(path + f).read())
                df_metrics = pd.io.json.json_normalize(metric["report"])
                del metric["report"]
                metrics.append(pd.DataFrame([metric]).join(df_metrics))
            except Exception as e:
                print(e)
        best_model = [f for f in files if "best" in f][0].split("_")[-3]
        metrics = pd.concat(metrics).drop_duplicates(subset=["model_pipeline_run"])
        metrics["best"] = metrics["model_pipeline_run"].apply(lambda x: 1 if x == best_model else 0)
        metrics = metrics.sort_values("test_score", ascending=False)
        return metrics

    def explain(self, data, shap_exist=False, global_explain=True, top_n=20):

        model, model_name, best_metrics = self.get_best_model(True, True)
        model_name = model_name.split("_")[-1]
        target = self.problem
        path = "automl/explain/{}/".format(target)
        if not os.path.exists(path):
            os.mkdir(path)
        df_metrics = self.get_metrics_of_models()
        df_metrics.to_csv(path + "metrics.csv", index=False)
        df_metrics.drop("columns", axis=1).to_csv(path + "metrics_no_cols.csv", index=False)
        new_cols = {}
        features = best_metrics["columns"]
        features_new = []
        for col in features:
            try:
                t = col.split("_")[-1].split("(")[-1]
                if "t-" in col:
                    new_col = "{}*{}({}".format(col.split("-")[0][:-1], col.split("-")[1], t)
                else:
                    new_col = "{}*{}".format(col.split("-")[0][:-1], col.split("-")[1])
            except Exception as e:
                new_col = col
            new_cols[col] = new_col
            features_new.append(new_col)

        if model_name in ["rf", "xgboost"]:
            importances = model.best_estimator_.steps[-1][-1].feature_importances_
            indices = np.argsort(importances)[-top_n:]
            plt.figure()
            plt.title('Feature Importances')
            plt.yticks(range(len(indices)), [features_new[i] for i in indices], rotation=15, stretch=500)
            plt.xlabel('Relative Importance')
            plt.barh(range(len(indices)), importances[indices], color='b')
            plt.savefig(path + "importances.png", bbox_inches="tight")
            # plt.show()
            indices = indices[::-1]
            df = pd.DataFrame([(a, b) for a, b in zip(importances[indices], [features_new[i] for i in indices])],
                              columns=["importance", "feature"])
            if not os.path.exists(path):
                os.mkdir(path)
            df.to_csv(path + model_name + "_importances.csv", index=False)

        if shap_exist:
            try:
                shap_val = pickle.load(open(path + "shap_val_{}.p".format(model_name), "rb"))
            except Exception as e:
                print(e)
        else:
            if model_name in ["rf", "xgboost"]:
                ex = shap.TreeExplainer(model.best_estimator_.steps[-1][1])
            else:
                ex = shap.KernelExplainer(model.best_estimator_.steps[-1][1])
            try:
                shap_val = ex.shap_values(data[features].values)
                if len(shap_val) == 2:
                    shap_val = shap_val[-1]
                    expected_value = ex.expected_value[-1]
                else:
                    expected_value = ex.expected_value
            except Exception as e:
                shap_val = ex.shap_values(data[features].values)
            pickle.dump(shap_val, open(path + "shap_val_{}.p".format(model_name), "wb"))

        data = data.rename(columns=new_cols)
        features = features_new
        local_shap_df = self.shap_analysis(data, shap_val, features, data.columns[0], data[data.columns[0]].unique().tolist())
        local_shap_df.to_csv(path + "local_shap_df.csv", index=False)
        if global_explain:
            plt.figure()
            shap.summary_plot(shap_val, data[features], show=False)
            plt.savefig(path + "shap_summary_plot.png", bbox_inches="tight")
            # plt.show()
            # shap.force_plot(expected_value, shap_val, data[features], show=False)
            # plt.savefig(path + "shap_force_plot.png", bbox_inches="tight")
            # plt.show()
            columns = features
            index = 0
            corr_index = 0
            columns_short = columns[index:]
            # columns_short = [col for col in columns_short if "SUM_OF_INTERNET_YEAR_ADVANCED_INCOME" not in col.upper() and "SUM_OF_PPA_INCOME" not in col.upper()]
            columns_short_index = []
            for i, v in enumerate(columns):
                if v in columns_short:
                    columns_short_index.append(i)
            # abs_sum = np.abs(shap_val[:, columns_short_index]).sum(axis=(0, 1))
            # regular_sum = shap_val[:, columns_short_index].sum(axis=(0, 1))
            # mean_ = shap_val[:, columns_short_index].mean(axis=(0, 1))
            abs_sum = np.abs(shap_val[:, columns_short_index]).sum(axis=(0))
            regular_sum = shap_val[:, columns_short_index].sum(axis=(1))
            mean_ = shap_val[:, columns_short_index].mean(axis=(1))
            arg_min = abs_sum.argmin()
            arg_max = abs_sum.argmax()
            best_min = columns_short[arg_min]
            best_max = columns_short[arg_max]
            idx = (-abs_sum).argsort()[:top_n]
            # cols_s = [col.split("_VERTICAL_ID")[0] for col in columns_short]
            cols_s = columns_short
            best_features_sum = [abs_sum[i] for i in idx]
            best_features_regular_sum = [regular_sum[i] for i in idx]
            best_features_mean = [mean_[i] for i in idx]
            best_features = [cols_s[i] for i in idx]
            ############################################ check ######################################
            corrs = {}
            for i in idx:
                corr = np.corrcoef(shap_val[:, columns_short_index][:, i], data[features].values[:, columns_short_index][:, i])
                # corr = np.corrcoef(shap_val[:, columns_short_index][:, i].sum(axis=1),
                #                    data[:, columns_short_index][:, i].sum(axis=1))
                corrs[cols_s[i]] = corr[0, 1] if not np.isnan(corr[0, 1]) else 0
            plt.figure()
            ax = pd.Series(best_features_sum, index=best_features).plot(kind='barh',
                                                                        color=['red' if corr > 0 else 'blue' for
                                                                               corr in corrs.values()],
                                                                        x='Variable', y='SHAP_abs')
            ax.set_xlabel("SHAP Value (Red = Positive Impact) model: {} target: {}".format(model_name, target))
            plt.savefig(path + "shap_global_explain.png", bbox_inches="tight")
            # plt.show()

@staticmethod
def _plot_roc(model, y, score):

    plt.figure()
    lw = 2
    fpr, tpr, _ = metrics.roc_curve(y, score)
    roc_auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve {}'.format(model))
    plt.legend(loc="lower right")
    plt.show()
