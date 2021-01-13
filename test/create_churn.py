from automl.dev_tools import get_cols
import pandas as pd
import random
import json
import numpy as np


def churn_feature(s, decay, month_alpha):

    b = s.copy(True)
    if s.dtype == "O":
        return s
    x = max([len(s) - month_alpha, 0])
    y = max([len(s) - month_alpha, 0])
    a = y + decay * np.exp(decay * len(s))
    for i in range(x, len(s)):
        s.iloc[i] = max([a - decay * np.exp(decay * (i + 1)), 0])
    # print(s)
    # print(b)
    # print(1)
    return s


def churn(x):

    x = np.zeros(x.shape[0])
    x[-1] = 1
    return x


if __name__ == "__main__":

    sample_size = 0.01
    cols_sample = 0.1
    month_alpha = 2
    decay = 0.1
    target_col = "target_churn"
    params = json.loads(open("test/params.json", "rb").read())
    key_cols = params["key_cols"]

    df = pd.read_csv('test/df_joined_old.csv')
    df = df.sort_values(key_cols, ascending=True)
    columns = get_cols(df)
    columns = columns["numeric"] + columns["categoric"]
    accounts = df[df["target_loan"] == 0]["account_id"].drop_duplicates()
    accounts = accounts.sample(int(accounts.shape[0] * sample_size)).to_list()
    len(accounts)
    columns = [col for col in columns if random.random() < cols_sample]
    df_no_churn = df[~df["account_id"].isin(accounts)]
    df_no_churn[target_col] = 0
    df_churn = df[df["account_id"].isin(accounts)]
    dfs = []
    for account in accounts:
        df_c = df_churn[df_churn["account_id"] == account]
        for col in columns:
            df_c[col] = df_c.groupby("account_id")[col].transform(lambda x: churn_feature(x, decay, month_alpha))
        df_c[target_col] = df_c.groupby("account_id")[key_cols[1]].transform(lambda x: churn(x))
        dfs.append(df_c)
    df_churn = pd.concat(dfs)
    check = df_churn[key_cols + columns]
    df = pd.concat([df_no_churn, df_churn])
    df.to_csv('test/df_joined.csv', index=False)
    print(1)
