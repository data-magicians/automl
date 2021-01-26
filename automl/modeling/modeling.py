import sys
sys.path.append("automl")
from automl.dev_tools import *
from automl.preprocessing.transformers import *
from automl.preprocessing.pipelines import *
from sklearn.model_selection import RandomizedSearchCV, KFold
import time
import pickle
import gc
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from sklearn import metrics
import re
from sklearn.preprocessing import LabelEncoder
import shap
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import svm
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
from xgboost import XGBRegressor, XGBClassifier
# from keras import backend as backend


def model_pipeline_run_unpack(args):
    """
    used to unpack and to run in multiprocessing
    :param args: the arguments to pass - tuple
    """
    to_return = model_pipeline_run(*args)
    del args
    gc.collect()
    return to_return


def scoring(model, X_train, X_test, y_train, y_test, columns, row={}, model_name="", type=""):
    """

    :param model:
    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :param columns:
    :param row:
    :param model_name:
    :param type:
    :return:
    """
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    row["model_pipeline_run"] = model_name
    row["hyperparameters"] = model.best_params_
    row["train_score"] = train_score
    row["test_score"] = test_score
    try:
        row["train_y_perc"] = y_train.mean()[0]
    except Exception as e:
        row["train_y_perc"] = y_train.mean()
    try:
        row["test_y_perc"] = y_test.mean()[0]
    except Exception as e:
        row["test_y_perc"] = y_test.mean()
    row["train_n"] = X_train.shape[0]
    row["train_m"] = X_train.shape[1]
    row["test_n"] = X_test.shape[0]
    row["test_m"] = X_test.shape[1]
    row["columns"] = "*|*".join(columns)
    row["grid"] = model
    row["type"] = type
    y_pred = model.predict(X_test)
    train_y_pred = model.predict(X_train)
    if type == "classification":
        target_names = sorted(list(set(y_test)))
        report_train = metrics.classification_report(y_train, train_y_pred, labels=target_names, output_dict=True)
        report_train = {k + "_train": v for k, v in report_train.items()}
        report_test = metrics.classification_report(y_test, y_pred, labels=target_names, output_dict=True)
        report_test = {k + "_test": v for k, v in report_test.items()}
        report = dict(**report_train, **report_test)
        try:
            # train
            fpr, tpr, thresholds = metrics.roc_curve(y_train, model.predict(X_train), pos_label=1)
            report["auc_train"] = metrics.auc(fpr, tpr)
            predict_proba = model.predict_proba(X_train)
            ind = predict_proba.shape[-1] - 1
            p, r, th = metrics.precision_recall_curve(y_train, predict_proba[:, ind])
            report["best_t_train"] = best_t(p, r, th)
            report["train_precisions"] = p
            report["train_recalls"] = r
            report["train_threshold_pr"] = th
            report["train_threshold_roc"] = thresholds
            report["train_tpr"] = tpr
            report["train_fpr"] = fpr
            # test
            #todo change this to prediction using the best t found on train, and deal with multiclass aswell
            fpr, tpr, thresholds = metrics.roc_curve(y_test, model.predict(X_test), pos_label=1)
            report["auc_test"] = metrics.auc(fpr, tpr)
            predict_proba = model.predict_proba(X_test)
            ind = predict_proba.shape[-1] - 1
            p, r, th = metrics.precision_recall_curve(y_test, predict_proba[:, ind])
            report["best_t_test"] = best_t(p, r, th)
            report["test_precisions"] = p
            report["test_recalls"] = r
            report["test_threshold_pr"] = th
            report["test_threshold_roc"] = thresholds
            report["test_tpr"] = tpr
            report["test_fpr"] = fpr
            report["cm_train"] = metrics.confusion_matrix(y_train, train_y_pred).tolist()
            report["cm_test"] = metrics.confusion_matrix(y_test, y_pred).tolist()
        except Exception as e:
            print(e)
    else:
        report = {"test_r2_score": metrics.r2_score(y_test, y_pred),
                  "test_r2_score_adj": 1 - ((1 - metrics.r2_score(y_test, y_pred)) * (row["test_n"] - 1) / (row["test_n"] - row["test_m"] - 1)),
                  "test_median_absolute_error": metrics.median_absolute_error(y_test, y_pred),
                  "test_mse": metrics.mean_squared_error(y_test, y_pred),
                  "test_rmse": metrics.mean_squared_error(y_test, y_pred) ** 0.5,
                  "test_mean_absolute_error": metrics.mean_absolute_error(y_test, y_pred),
                  "test_explained_variance_score": metrics.explained_variance_score(y_test, y_pred),
                  "test_nrmse": np.power(metrics.mean_squared_error(y_test, y_pred), 0.5) / np.power(y_test.max() - y_test.min(), 2),
                  "test_cv_rmse": (metrics.mean_squared_error(y_test, y_pred) ** 0.5) / row["test_y_perc"],
                  "test_cv_mae": metrics.mean_absolute_error(y_test, y_pred) / row["test_y_perc"],
                  "train_cv_rmse": (metrics.mean_squared_error(y_train, train_y_pred) ** 0.5) / row["train_y_perc"],
                  "train_cv_mae": metrics.mean_absolute_error(y_train, train_y_pred) / row["train_y_perc"],
                  "train_r2_score": metrics.r2_score(y_train, train_y_pred),
                  "train_r2_score_adj": 1 - ((1 - metrics.r2_score(y_train, train_y_pred)) * (row["train_n"] - 1) / (row["train_n"] - row["train_m"] - 1)),
                  "train_median_absolute_error": metrics.median_absolute_error(y_train, train_y_pred),
                  "train_mse": metrics.mean_squared_error(y_train, train_y_pred),
                  "train_rmse": metrics.mean_squared_error(y_train, train_y_pred) ** 0.5,
                  "train_mean_absolute_error": metrics.mean_absolute_error(y_train, train_y_pred),
                  "train_explained_variance_score": metrics.explained_variance_score(y_train, train_y_pred),
                  "train_nrmse": np.power(metrics.mean_squared_error(y_train, train_y_pred), 0.5) / np.power(
                      y_train.max() - y_train.min(), 2)}
    row["report"] = report
    print("finished score for: {}".format(row["dataset_index"]))
    return row


def over_sample(X_train, y_train):

    list_index = []
    X_train_new = []
    y_train_new = []
    for i, v in enumerate(y_train):
        if v > 0:
            list_index.append(i)
    total_sample = X_train.shape[0] - 2 * len(list_index)
    try:
        for i in range(total_sample):
            n = list_index[np.random.randint(0, len(list_index), size=1)[0]]
            X_train_new.append(X_train.iloc[n])
            y_train_new.append(y_train.iloc[n])
        return pd.concat([X_train, pd.DataFrame(X_train_new)]), pd.concat([y_train, pd.Series(y_train_new)])
    except Exception as e:
        print(e)
        print("the error is in oversample")
        return X_train, y_train


def model_pipeline_run(index, model, params, X_train, y_train, X_test, y_test, model_name, pre_process_time, type):
    """
    running the model_pipeline_run using a pipeline and with a grid search
    :param index: index of the data from all the
    :param model: the classifier/regressor to use
    :param params: the hyperparameters to optimize
    :param X_train: the train dataset
    :param y_train: the train target
    :param X_test: the test dataset
    :param y_test: the test target
    :return: a row in the information about run table
    """
    n_jobs = -1
    n_iter = 100
    if model is None:
        return
    try:
        row = {"dataset_index": index}
        if type == "classification":
            steps = [("classifier", model)]
        else:
            steps = [("regressor", model)]
        pipeline = MLPipeline(steps=steps)
        if type == "classification":
            if model_name == "rf":
                params["classifier__max_features"] = [min([x, X_train.shape[1]]) for x in
                                                      params["classifier__max_features"]]
            elif "dl" in model_name:
                n_jobs = None
                params["classifier__shape"] = [X_train.shape[1]]
            if isinstance(y_test, (str)):
                try:
                    y_train = np.asarray(list(map(lambda x: int(re.search("[0-9]+", x).group()), y_train)))
                    y_test = np.asarray(list(map(lambda x: int(re.search("[0-9]+", x).group()), y_test)))
                except Exception as e:
                    le = LabelEncoder()
                    y_train = le.fit_transform(y_train)
                    y_test = le.transform(y_test)
            grid = RandomizedSearchCV(estimator=pipeline, param_distributions=params, cv=KFold(3), refit=True,
                                      verbose=0, n_jobs=n_jobs, n_iter=n_iter,
                                      scoring="f1" if len(set(y_train)) == 2 else "f1_weighted")
        else:
            if model_name == "rf":
                params["regressor__max_features"] = [min([x, X_train.shape[1]]) for x in
                                                     params["regressor__max_features"]]
            elif "dl" in model_name:
                n_jobs = None
                params["regressor__shape"] = [X_train.shape[1]]
            grid = RandomizedSearchCV(estimator=pipeline, param_distributions=params, cv=KFold(3), refit=True,
                                      verbose=0, n_jobs=n_jobs, n_iter=n_iter, error_score=np.nan)
        model_time = time.time()
        columns = X_train.columns
        if "dl-rnn" in model_name:
            X_train = np.reshape(X_train.astype("float32").values, (X_train.shape[0], 1, X_train.shape[1]))
            X_test = np.reshape(X_test.astype("float32").values, (X_test.shape[0], 1, X_test.shape[1]))
        elif "xgb" in model_name:
            X_train = pd.DataFrame(X_train.astype("float32").values, columns=columns)
            X_test = pd.DataFrame(X_test.astype("float32").values, columns=columns)
        else:
            X_train = X_train.astype("float32").values
            X_test = X_test.astype("float32").values
        grid = grid.fit(X_train, y_train)
        row["time"] = (time.time() - model_time) / 60
        row["pre_process_time"] = pre_process_time
        return scoring(grid, X_train, X_test, y_train, y_test, columns, row=row, model_name=model_name, type=type)
    except Exception as e:
        print(e)


def deeplearning(shape, dropout=0.5, lr=0.1, l1=0.1, l2=0.1, n_hidden_layers=3, num_neurons=512,
                 num_classes=1, type="classification", activation="relu"):
    """
    :param shape:
    :param dropout:
    :param lr:
    :param l1:
    :param l2:
    :param n_hidden_layers:
    :param num_neurons:
    :param num_classes:
    :param type:
    :param activation:
    :param rnn:
    :return:
    """
    print_summary = False
    optimizer_m = optimizers.Adam(lr=lr)
    model = models.Sequential()

    model.add(layers.Dense(num_neurons, activation=activation, input_shape=(shape,)))
    # Hidden - Layers
    for i in range(n_hidden_layers):
        model.add(layers.Dense(num_neurons, activation=activation, name='fc{}'.format(i)))
        model.add(layers.Dropout(dropout))
    # Output- Layer
    if type == "classification":
        if num_classes == 1:
            model.add(layers.Dense(num_classes, activation="sigmoid", name="output"))
        else:
            model.add(layers.Dense(num_classes, activation='softmax', name="output",
                                   kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
    else:
        model.add(layers.Dense(num_classes, activation='linear', name="output",
                               kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
    # compiling the model_pipeline_run
    if type == "classification":
        model.compile(optimizer=optimizer_m, loss="binary_crossentropy", metrics=[f1_m])
    else:
        model.compile(optimizer=optimizer_m, loss="mean_squared_error", metrics=["mse"])
    if print_summary:
        try:
            print(model.summary())
        except Exception as e:
            print(e)
    return model


def deeplearning_rnn(shape, dropout=0.5, lr=0.1, l1=0.1, l2=0.1, n_hidden_layers=3, num_neurons=512,
                     num_classes=1, type="classification", activation="relu", periods_to_train=1, lstm_n=64):
    """
    :param shape:
    :param dropout:
    :param lr:
    :param l1:
    :param l2:
    :param n_hidden_layers:
    :param num_neurons:
    :param num_classes:
    :param type:
    :param activation:
    :param rnn:
    :return:
    """
    print_summary = False
    optimizer_m = optimizers.Adam(lr=lr)
    model = models.Sequential()
    model.add(layers.LSTM(lstm_n, input_shape=(periods_to_train, shape), return_sequences=True))
    # model.add(layers.Flatten())
    # model.add(layers.Dense(num_neurons, activation=activation, input_shape=(shape,)))
    # Hidden - Layers
    for i in range(n_hidden_layers):
        model.add(layers.Dense(num_neurons, activation=activation, name='fc{}'.format(i)))
        model.add(layers.Dropout(dropout))
    # Output- Layer
    model.add(layers.Flatten())
    if type == "classification":
        if num_classes == 1:
            model.add(layers.Dense(num_classes, activation="sigmoid", name="output"))
        else:
            model.add(layers.Dense(num_classes, activation='softmax', name="output",
                                   kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
    else:
        model.add(layers.Dense(num_classes, activation='linear', name="output",
                               kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
    # compiling the model_pipeline_run
    if type == "classification":
        model.compile(optimizer=optimizer_m, loss="binary_crossentropy", metrics=[f1_m])
    else:
        model.compile(optimizer=optimizer_m, loss="mean_squared_error", metrics=["mse"])
    if print_summary:
        try:
            print(model.summary())
        except Exception as e:
            print(e)

    return model


def deeplearning_cnn(shape, dropout=0.5, lr=0.1, l1=0.1, l2=0.1, n_hidden_layers=3, num_neurons=512,
                     num_classes=1, type="classification", activation="relu", kernel_size=5, conv_stride=1, max_pool=2, pool_stride=2):
    """
    :param shape:
    :param dropout:
    :param lr:
    :param l1:
    :param l2:
    :param n_hidden_layers:
    :param num_neurons:
    :param num_classes:
    :param type:
    :param activation:
    :param rnn:
    :return:
    """
    print_summary = False
    optimizer_m = optimizers.Adam(lr=lr)
    model = models.Sequential()
    model.add(layers.Conv2D(num_neurons, kernel_size=(kernel_size, kernel_size), strides=(conv_stride, conv_stride), activation=activation, input_shape=(shape,)))
    model.add(layers.MaxPooling2D(pool_size=(max_pool, max_pool), strides=(pool_stride, pool_stride)))
    model.add(layers.Flatten())
    # Hidden - Layers
    for i in range(n_hidden_layers):
        model.add(layers.Dense(num_neurons, activation=activation, name='fc{}'.format(i)))
        model.add(layers.Dropout(dropout))
    # Output- Layer
    model.add(layers.Flatten())
    if type == "classification":
        if num_classes == 1:
            model.add(layers.Dense(num_classes, activation="sigmoid", name="output"))
        else:
            model.add(layers.Dense(num_classes, activation='softmax', name="output",
                                   kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
    else:
        model.add(layers.Dense(num_classes, activation='linear', name="output",
                               kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
    # compiling the model_pipeline_run
    if type == "classification":
        model.compile(optimizer=optimizer_m, loss="binary_crossentropy", metrics=[f1_m])
    else:
        model.compile(optimizer=optimizer_m, loss="mean_squared_error", metrics=["mse"])
    if print_summary:
        try:
            print(model.summary())
        except Exception as e:
            print(e)
    return model


def best_t(precisions, recalls, thresholds):
    """
    calculate the best threshold by F1 measure
    :param precisions: precisions from the precision-recall curve - list of float
    :param recalls: recalls from the precision - recall curve - list of float
    :param thresholds: thresholds from the precision-recall curve - list of float
    :return: the best threshold - float
    """
    f1 = [2 * (precisions[i] * recalls[i]) / (precisions[i] + recalls[i]) for i in range(0, len(thresholds))]
    return thresholds[np.argmax(f1)]
