from datetime import datetime
import pandasql as ps
from datetime import timedelta
import time
from automl.preprocessing.transformers import *
from automl.preprocessing.pipelines import *
import logging
from automl.dev_tools import *


logging.basicConfig(format='%(asctime)s     %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    stream=sys.stdout,
                    level='INFO')


def get_data():

    raw_data_files = os.listdir(os.getcwd()+"/raw_data")
    raw_data_files_clean = [file.split(".")[0] for file in raw_data_files]
    data = {}
    for name in raw_data_files_clean:
        data[name] = pd.read_csv(os.getcwd()+"/raw_data/"+name+".csv",sep=";")
    # strip spaces from all columns names (left and right)

    for key in list(data.keys()):
        clean_col_names = [name.strip() for name in data[key].columns]
        data[key].columns = clean_col_names

    return data


def int_to_date(date):

    if type(date)==str or ~np.isnan(date):
        date = str(date)
        date = datetime(year=int("19"+date[0:2]), month=int(date[2:4]), day=int(date[4:6]))

    return date


def create_dates_table(start_date,end_date):
    
    # let's create a date table:
    # expected format is: "1997-12-31"
    # since 2018, all the mondays dates
    # I only use customers that have at least 3 month of data
    # But i need to create descriptives to say if it for example 1 month, 2 month, 3 month. Need to understand behaviour of new members
    date_df = pd.DataFrame({'validity_date_start_month': pd.date_range(start_date, end_date, freq="MS")})
    #print(date_df.shape)
    date_df = pd.concat([date_df,pd.DataFrame({'validity_date_end_month': pd.date_range(start_date, end_date, freq="M")})],axis=1)
    #print(date_df.shape)
    date_df["monthtable_rownumber"] = date_df.index+1
    date_df["monthtable_rownumber"] = date_df["monthtable_rownumber"].astype(int)

    return date_df


def feature_engineering_transactions(df_trans, dates_df):

    df_trans["date"] = df_trans["date"].apply(lambda x: int_to_date(x))
    sqlcode = '''
    select *
    from dates_df
    inner join df_trans on dates_df.validity_date_start_month <= df_trans.date and dates_df.validity_date_end_month >= df_trans.date
    '''
    trans_df = ps.sqldf(sqlcode,locals())
    #print(trans_df.shape)
    trans_df["validity_date_start_month"] = trans_df["validity_date_start_month"].str[:10]
    trans_df["validity_date_end_month"] = trans_df["validity_date_end_month"].str[:10]
    trans_df = trans_df.sort_values(by=["account_id","validity_date_start_month","monthtable_rownumber"])
    trans_df["date"] = trans_df["date"].str[:10]
    trans_df = trans_df.rename(columns={"date":"transaction_date"})
    trans_df = trans_df.sort_values(by=["account_id","monthtable_rownumber","trans_id"])
    # number of monthly transactions per account
    trans_df["count_month_alltrans_account"] = trans_df.groupby(['account_id','monthtable_rownumber'])["trans_id"].transform('count')
    # we can see we have a k-symbol col with no name (just blank string)
    trans_df.loc[trans_df["k_symbol"]==' ',"k_symbol"] = "other"
    # type field features
    type_cats = list(trans_df["type"].unique())
    for cat in type_cats:
        # count credit, total withdrawl, total cash withdrawl
        trans_df["count_month_"+cat] = np.where(trans_df["type"]==cat,1,0)
        trans_df["count_month_"+cat] = trans_df.groupby(['account_id','monthtable_rownumber'])["count_month_"+cat].transform('sum')
        # sum amount credit, total withdrawl, total cash withdrawl
        trans_df["sum_amount_month_"+cat] = np.where(trans_df["type"]==cat,trans_df["amount"],0)
        trans_df["sum_amount_month_"+cat] = trans_df.groupby(['account_id','monthtable_rownumber'])["sum_amount_month_"+cat].transform('sum')
        # avg amount credit, total withdrawl, total cash withdrawl
        trans_df["avg_amount_month_"+cat] = np.where(trans_df["type"]==cat,trans_df["amount"],0)
        trans_df["avg_amount_month_"+cat] = trans_df.groupby(['account_id','monthtable_rownumber'])["avg_amount_month_"+cat].transform('mean')
        trans_df["avg_amount_month_"+cat] = trans_df["avg_amount_month_"+cat].round(0)
    # ratio withdrawl/credit
    trans_df["ratio_month_wtdrl_credit"] = trans_df["sum_amount_month_withdrawal"]/trans_df["sum_amount_month_credit"]
    trans_df["ratio_month_wtdrl_credit"] = trans_df["ratio_month_wtdrl_credit"].round(2)
    # balance
    # avg month balance
    trans_df["avg_month_balance"] = trans_df.groupby(['account_id','monthtable_rownumber'])["balance"].transform('mean')
    # std inter-month balance
    trans_df["std_month_balance"] = trans_df.groupby(['account_id','monthtable_rownumber'])["balance"].transform('std')
    # if we have only one value it will return nan. will replace by 0
    trans_df["std_month_balance"] = trans_df["std_month_balance"].fillna(0)
    # min balance/ avg balance
    trans_df["min_month_balance"] = trans_df.groupby(['account_id','monthtable_rownumber'])["balance"].transform('min')
    trans_df["min_avg_ratio_month_balance"] = trans_df["min_month_balance"]/trans_df["avg_month_balance"]
    trans_df["min_avg_ratio_month_balance"] = trans_df["min_avg_ratio_month_balance"].round(2)
    # last balance of the month
    trans_df = trans_df.sort_values(by=["account_id","monthtable_rownumber","trans_id"])
    trans_df["max_acc_month_transid"] = trans_df.groupby(['account_id','monthtable_rownumber'])["trans_id"].transform('max')
    trans_df["endmonth_acc_balance"] = np.where(trans_df["max_acc_month_transid"]==trans_df["trans_id"],trans_df["balance"],np.nan)
    trans_df = trans_df.sort_values(by=["account_id","monthtable_rownumber","trans_id"])
    # total month withdrawl/end month balance
    trans_df["ratio_month_wtdrl_endbalance"] = trans_df["sum_amount_month_withdrawal"]/trans_df["endmonth_acc_balance"]
    trans_df["ratio_month_wtdrl_endbalance"] = trans_df["ratio_month_wtdrl_endbalance"].round(2)
    # credit+withdrawl/balance
    trans_df["ratio_month_wtdrl_cred_endbalance"] = (trans_df["sum_amount_month_withdrawal"]+trans_df["avg_amount_month_credit"])/trans_df["endmonth_acc_balance"]
    trans_df["ratio_month_wtdrl_cred_endbalance"] = trans_df["ratio_month_wtdrl_cred_endbalance"].round(2)
    # operation (vs type)
    operation_list = list(trans_df["operation"].value_counts().index)
    #print(operation_list)
    type_operation_dict = {'credit':['collection from another bank','credit in cash'], 'withdrawal':[cat for cat in operation_list if cat not in ['collection from another bank','credit in cash']]}
    #print(type_operation_dict)
    # features
    for key in list(type_operation_dict.keys()):
        for var in type_operation_dict[key]:
            trans_df["sum_amount_month_optype_"+key+"_"+var.replace(" ","_")] = np.where(trans_df["operation"]==var,trans_df["amount"],0)
            trans_df["sum_amount_month_optype_"+key+"_"+var.replace(" ","_")] = trans_df.groupby(['account_id','monthtable_rownumber'])["sum_amount_month_optype_"+key+"_"+var.replace(" ","_")].transform('sum')
            # I will not work with amounts, only with proportions
            trans_df["pct_sum_amount_month_optype_"+key+"_"+var.replace(" ","_")] = trans_df["sum_amount_month_optype_"+key+"_"+var.replace(" ","_")]/trans_df["sum_amount_month_"+key]
            trans_df["pct_sum_amount_month_optype_"+key+"_"+var.replace(" ","_")] = trans_df["pct_sum_amount_month_optype_"+key+"_"+var.replace(" ","_")].round(2)
            # if we dont have any at the relevant type we will recieve nan. I will impute at 0
            trans_df["pct_sum_amount_month_optype_"+key+"_"+var.replace(" ","_")] = trans_df["pct_sum_amount_month_optype_"+key+"_"+var.replace(" ","_")].fillna(0)
            del trans_df["sum_amount_month_optype_"+key+"_"+var.replace(" ","_")]
    # k_symbol
    k_symbol_list = list(trans_df["k_symbol"].value_counts().index)
    #print(k_symbol_list)
    type_k_symbol_dict = {'credit':['interest credited','old-age pension'], 'withdrawal':[cat for cat in k_symbol_list if cat not in ['interest credited','old-age pension']]}
    #print(type_k_symbol_dict)
    # features
    for key in list(type_k_symbol_dict.keys()):
        for var in type_k_symbol_dict[key]:
            trans_df["sum_amount_month_ktype_"+key+"_"+var.replace(" ","_")] = np.where(trans_df["k_symbol"]==var,trans_df["amount"],0)
            trans_df["sum_amount_month_ktype_"+key+"_"+var.replace(" ","_")] = trans_df.groupby(['account_id','monthtable_rownumber'])["sum_amount_month_ktype_"+key+"_"+var.replace(" ","_")].transform('sum')
            # I will not work with amounts, only with proportions
            trans_df["pct_sum_amount_month_ktype_"+key+"_"+var.replace(" ","_")] = trans_df["sum_amount_month_ktype_"+key+"_"+var.replace(" ","_")]/trans_df["sum_amount_month_"+key]
            trans_df["pct_sum_amount_month_ktype_"+key+"_"+var.replace(" ","_")] = trans_df["pct_sum_amount_month_ktype_"+key+"_"+var.replace(" ","_")].round(2)
            # if we dont have any at the relevant type we will recieve nan. I will impute at 0
            trans_df["pct_sum_amount_month_ktype_"+key+"_"+var.replace(" ","_")] = trans_df["pct_sum_amount_month_ktype_"+key+"_"+var.replace(" ","_")].fillna(0)
            del trans_df["sum_amount_month_ktype_"+key+"_"+var.replace(" ","_")]
    # collapse transactions to one obs per acc-month
    # for example just take last value of the month (doesnt matter anyway)
    trans_df = trans_df[trans_df["max_acc_month_transid"]==trans_df["trans_id"]]
    # delete vars that are irrelevant in the collapsed level
    vars_to_remove = ["transaction_date","type","operation","amount","balance","k_symbol","bank","account","trans_id","max_acc_month_transid"]
    for var in vars_to_remove:
        del trans_df[var]

    trans_df = trans_df.sort_values(by=["account_id","monthtable_rownumber"])
    # 2,3,4,6 month window avgs ma
    # because this is not a production level problem
    cols_for_ma_list = ["count_month_alltrans_account","count_month_credit","count_month_withdrawal","count_month_withdrawal in cash", "sum_amount_month_credit","sum_amount_month_withdrawal","sum_amount_month_withdrawal in cash", "ratio_month_wtdrl_credit", "std_month_balance","min_avg_ratio_month_balance","endmonth_acc_balance", "ratio_month_wtdrl_endbalance","ratio_month_wtdrl_cred_endbalance", "pct_sum_amount_month_optype_credit_collection_from_another_bank", "pct_sum_amount_month_optype_credit_credit_in_cash", "pct_sum_amount_month_optype_withdrawal_withdrawal_in_cash", "pct_sum_amount_month_optype_withdrawal_remittance_to_another_bank", "pct_sum_amount_month_optype_withdrawal_credit_card_withdrawal", "pct_sum_amount_month_ktype_credit_interest_credited", "pct_sum_amount_month_ktype_credit_old-age_pension", "pct_sum_amount_month_ktype_withdrawal_payment_for_statement", "pct_sum_amount_month_ktype_withdrawal_household","pct_sum_amount_month_ktype_withdrawal_other", "pct_sum_amount_month_ktype_withdrawal_insurrance_payment", "pct_sum_amount_month_ktype_withdrawal_sanction_interest_if_negative_balance"]
    for col in cols_for_ma_list:
        for period in [3,6]:
            trans_df['ma_month_'+str(period)+"_"+col] = trans_df.groupby(['account_id'])[col].transform(lambda x: x.rolling(period, period-1).mean())
    # ma ratios
    for col in cols_for_ma_list:
        trans_df["1m_3m_"+col] = trans_df[col]/trans_df["ma_month_3_"+col]-1
        trans_df["1m_3m_"+col] = trans_df["1m_3m_"+col].round(2)
        trans_df["1m_3m_"+col] = trans_df["1m_3m_"+col].fillna(0)
        # when we have 0/0 we get nan. I will impute it as 0
        trans_df["3m_6m_"+col] = trans_df["ma_month_3_"+col]/trans_df["ma_month_6_"+col]-1
        trans_df["3m_6m_"+col] = trans_df["3m_6m_"+col].round(2)
        # when we have 0/0 we get nan. I will impute it as 0
        trans_df["3m_6m_"+col] = trans_df["3m_6m_"+col].fillna(0)
    #print(trans_df.shape)
    return trans_df


def clean_and_join_other_data(data):
    
    # join client and disp
    #print(data["client"].shape)
    #print(data["disp"].shape)
    df = pd.merge(data["client"],data["disp"],on="client_id",how="inner")
    #print(df.shape)
    # join account meta to the client level and get account-client as key
    #print(df.shape)
    #print(data["account"].shape)
    #print(data["account"].head())
    df = pd.merge(df,data["account"],on="account_id",how="inner")
    # remove key field duplication and rename back as before the join
    del df["district_id_y"]
    df = df.rename(columns={"district_id_x":"district_id"})
    # also rename arbitrary name of field "date" which exists in multiple tables to the relevant table it came from
    df = df.rename(columns={"date":"account_date"})
    #print(df.shape)
    # merge district level data (no date relations it is just general meta data on the district)
    # clean question marks in two numeric cols, impute them with mean
    mean_unemp95 = data["district"][data["district"]["unemploymant rate '95"]!="?"]["unemploymant rate '95"].astype(float).mean().round(2)
    mean_crime95 = data["district"][data["district"]["no. of commited crimes '95"]!="?"]["no. of commited crimes '95"].astype(float).mean().round(0)
    #print(mean_unemp95)
    #print(mean_crime95)
    data["district"].loc[data["district"]["unemploymant rate '95"]=="?","unemploymant rate '95"] = mean_unemp95
    data["district"].loc[data["district"]["no. of commited crimes '95"]=="?","no. of commited crimes '95"] = mean_crime95
    # set handled fields with relevant dtype setting
    data["district"]["unemploymant rate '95"] = data["district"]["unemploymant rate '95"].astype(float)
    data["district"]["no. of commited crimes '95"] = data["district"]["no. of commited crimes '95"].astype(float)
    #print(df.shape)
    #print(data["district"].shape)
    df = pd.merge(df,data["district"],left_on="district_id",right_on="code",how="inner")
    df = df.rename(columns={"name":"district_name"})
    #print(df.shape)
    # at this point we have vars from the future (commited crimes in 96 when account opned in 95)
    # but we will not keep our table at this level
    # will handle that once we get to the monthly panel
    # some cleaning and features
    # handling birth_number and extracting gender
    # Extracts all genders from birthdate pattern (for women month = month_numer + 50; for men month = month_number)
    # Get the 3rd and 4th character from "birth_number". If it is > 12
    # that row is for Female, otherwise Male
    df["gender"] = np.where(df["birth_number"].astype(str).str[2:4].astype(int)>12,"female","male")
    # Now correct the "birth_number". Subtract 50 form middle 2 digits.
    # Updated based on feedback from @RuiBarradas to use df$Gender == "Female" 
    # to subtract 50 from month number
    df["birth_number"] = np.where(df["gender"]=="female",df["birth_number"]-5000,df["birth_number"])
    # convert to regular date
    df["birth_number"] = df["birth_number"].apply(lambda x: int_to_date(x))
    df["account_date"] = df["account_date"].apply(lambda x: int_to_date(x))
    # per account features
    # number of clients in account
    df["num_clients_account"] = df.groupby('account_id')["client_id"].transform('count')
    # num districts in account
    df["num_districts_account"] = df.groupby('account_id')["district_id"].transform('nunique')
    # num genders in account
    df["num_genders_account"] = df.groupby('account_id')["gender"].transform('nunique')
    # district features
    # pct change unemployment 95-96 (can only be used in 96)
    df["pctchg_unempl_9695"] = df["unemploymant rate '96"]/df["unemploymant rate '95"]-1
    df["pctchg_unempl_9695"] = df["pctchg_unempl_9695"].round(1)
    # pct change crimes 95-96 (can only be used in 96)
    df["pctchg_crime_9695"] = df["no. of commited crimes '96"]/df["no. of commited crimes '95"]-1
    df["pctchg_crime_9695"] = df["pctchg_crime_9695"].round(1)
    # cards
    data["card_train"]["issued"] = data["card_train"]["issued"].apply(lambda x: int_to_date(x))
    data["card_train"] = data["card_train"].rename(columns={"type":"card_type","issued":"card_issued"})
    # join cards
    #print(df.shape)
    df = pd.merge(df,data["card_train"][["disp_id","card_type","card_issued"]],how="left",on="disp_id")
    #print(df.shape)
    return df


def join_data_final(df,trans_df,data):

    # join meta data df to trans
    # because df is in client level, trans_df will be larger now
    #print(trans_df.shape)
    #print(df.shape)
    df_joined = pd.merge(trans_df,df,on="account_id",how="left")
    #print(df_joined.shape)
    # to bring it to account level I just need to handle ages of clients
    df_joined["client_age"] = (df_joined["validity_date_end_month"].astype('datetime64[ns]') - df['birth_number'])/timedelta(days=365)
    df_joined["client_age"] = df_joined["client_age"].round(0)
    # impute mean
    df_joined["client_age"] = df_joined["client_age"].fillna(df_joined["client_age"].mean()).round(0)
    # min age acc, max age acc
    df_joined["max_age_acc"] = df_joined.groupby(['account_id','monthtable_rownumber'])["client_age"].transform('max')
    df_joined["min_age_acc"] = df_joined.groupby(['account_id','monthtable_rownumber'])["client_age"].transform('min')
    # and also the card vars which are in disp level
    # in case of datetime fields nan is NaT, cannot add regular nan np.datetime64('2088-01-01'), np.datetime64(NaT)
    # I will assume arbitrary date atm
    df_joined["card_type"] = np.where( (df_joined["card_issued"]>df_joined["validity_date_end_month"].astype('datetime64[ns]')) | (df_joined["card_type"].isna()), "no_card",df_joined["card_type"])
    df_joined["years_since_card_issued"] = np.where(df_joined["card_type"]=="no_card", -100.0, (df_joined["validity_date_end_month"].astype('datetime64[ns]') - df_joined['card_issued'])/timedelta(days=365))
    df_joined["years_since_card_issued"] = df_joined["years_since_card_issued"].round(2)
    # if one of the disps has a card and the other doesn't I will take the max of -100 and something positive
    df_joined["years_since_card_issued"] = df_joined.groupby(['account_id','monthtable_rownumber'])["years_since_card_issued"].transform('max')
    # no need for account date
    df_joined["years_since_account_date"] = (df_joined["validity_date_end_month"].astype('datetime64[ns]') - df_joined['account_date'])/timedelta(days=365)
    df_joined["years_since_account_date"] = df_joined["years_since_account_date"].round(2)

    # 95 96 vars make sure district features with years are only used in the panel in releveant dates and avoid leakage
    df_joined["unemploymant rate '95"] = np.where(np.datetime64('1995-01-01')<=df_joined["validity_date_end_month"].astype('datetime64[ns]'), df_joined["unemploymant rate '95"], -100.0)
    df_joined["unemploymant rate '96"] = np.where(np.datetime64('1996-01-01')<=df_joined["validity_date_end_month"].astype('datetime64[ns]'), df_joined["unemploymant rate '96"], -100.0)
    df_joined["no. of commited crimes '95"] = np.where(np.datetime64('1995-01-01')<=df_joined["validity_date_end_month"].astype('datetime64[ns]'), df_joined["no. of commited crimes '95"], -100.0)
    df_joined["no. of commited crimes '96"] = np.where(np.datetime64('1996-01-01')<=df_joined["validity_date_end_month"].astype('datetime64[ns]'), df_joined["no. of commited crimes '96"], -100.0)
    df_joined["pctchg_unempl_9695"] = np.where(np.datetime64('1996-01-01')<=df_joined["validity_date_end_month"].astype('datetime64[ns]'), df_joined["pctchg_unempl_9695"], -100.0)
    df_joined["pctchg_crime_9695"] = np.where(np.datetime64('1996-01-01')<=df_joined["validity_date_end_month"].astype('datetime64[ns]'), df_joined["pctchg_crime_9695"], -100.0)
    accounts_with_cards = df_joined[df_joined["card_type"]!="no_card"].sort_values(by=["account_id","monthtable_rownumber","card_type"])["account_id"].tolist()
    # bring it to account level
    # gender will be owner gender
    df_joined = df_joined[df_joined["type"]=="OWNER"]
    #print(df_joined.shape)
    vars_to_remove_joined = ["client_id","disp_id","type","birth_number","code","card_issued","client_age","account_date"]

    for var in vars_to_remove_joined:
        del df_joined[var]

    df_joined = df_joined.drop_duplicates()
    #print(df_joined.shape)
    # join the loan data
    data["loan_train"]["date"] = data["loan_train"]["date"].apply(lambda x: int_to_date(x))
    data["loan_train"] = data["loan_train"].rename(columns={"date":"loan_date"})
    #print(df_joined.shape)
    #print(data["loan_train"].shape)
    df_joined = pd.merge(df_joined,data["loan_train"],on="account_id",how="left")
    #print(df_joined.shape)
    df_joined["target_loan"] = np.where(~(df_joined["loan_id"].isna()),1,0)
    df_joined["target_amount"] = np.where(~(df_joined["loan_id"].isna()),df_joined["amount"],0)
    # also make the target 0 if loan date didnt arrive yet
    df_joined["target_loan"] = np.where( (df_joined["loan_date"]>df_joined["validity_date_end_month"].astype('datetime64[ns]'))
                                     | (df_joined["card_type"].isna()), 0,df_joined["target_loan"])
    df_joined["target_amount"] = np.where( (df_joined["loan_date"]>df_joined["validity_date_end_month"].astype('datetime64[ns]'))
                                     | (df_joined["card_type"].isna()), 0,df_joined["target_amount"])
    del df_joined["loan_id"]
    del df_joined["amount"]
    del df_joined["loan_date"]
    del df_joined["duration"]
    del df_joined["payments"]
    del df_joined["status"]
    return df_joined


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
        logging.info(e)
        print("the error is in oversample")
        return X_train, y_train


def features_pipeline(index, X_train, y_train ,X_test, y_test, columns, row, spark, key=None, date=None, static_cols=[],
                      r=1, w=2, corr_per=0.7, oversample=True):
    """
    running a problemread_data
    :param index: the index of the sampled from the original dataset - int
    :param df: the learning dataset to perform explain over - Dataframe
    :param X_test: the test dataset - Dataframe
    :param y_test: the test target - Dataframe
    :param columns: the columns dictionary - dictionary where each key has list of features names - dictionary
    :return: a list of one tuple with results and information about the model_pipeline_run run
    """
    start_time = time.time()
    # X_train = df.drop(columns["target"][0], axis=1)
    # y_train = df[columns["target"][0]]
    file_name = "preprocess_results/preprocess_pipeline_{}".format(row["target"])
    # pre proccess pipeline stages
    clear_stage = ClearNoCategoriesTransformer(categorical_cols=columns["categoric"])
    imputer = ImputeTransformer(numerical_cols=columns["numeric"], categorical_cols=columns["categoric"],
                                strategy="time_series", key_field=key, date_field=date, parallel=True)
    outliers = OutliersTransformer(numerical_cols=columns["numeric"], categorical_cols=columns["categoric"])
    scale = ScalingTransformer(numerical_cols=columns["numeric"])
    if row["type"] == "classification":
        categorize = CategorizeByTargetTransformer(categorical_cols=columns["categoric"])
    else:
        categorize = CategorizingTransformer(categorical_cols=columns["categoric"])
    chisquare = ChiSquareTransformer(categorical_cols=columns["categoric"], numerical_cols=columns["numeric"])
    correlations = CorrelationTransformer(numerical_cols=columns["numeric"], categorical_cols=columns["categoric"],
                                          target=columns["target"], threshold=corr_per)
    dummies = DummiesTransformer(columns["categoric"])
    timeseries = TimeSeriesTransformer(key=key, date=date, key_col=X_train.reset_index()[key], date_col=X_train.reset_index()[date],
                                       split_y=False, static_cols=static_cols, r=r, w=w, target=columns["target"][0])
    steps_feat = [("clear_non_variance", clear_stage),
                  ("imputer", imputer),
                  ("outliers", outliers),
                  ("scaling", scale),
                  ("chisquare", chisquare),
                  ("correlations", correlations),
                  ("categorize", categorize),
                  ("dummies", dummies),
                  ("timeseries", timeseries)]
    if key is None:
        steps_feat = steps_feat[:-1]
    pipeline_feat = Pipeline(steps=steps_feat)
    pipeline_feat = pipeline_feat.fit(X_train, y_train)
    X_train = pipeline_feat.transform(X_train)
    X_test = pipeline_feat.transform(X_test)
    finish_time = time.time()
    time_in_minutes = (finish_time - start_time) / 60
    if not os.path.exists("preprocess_results/"):
        os.mkdir("preprocess_results/")
    save(open(file_name, "wb"), (row["target"], index, time_in_minutes, pipeline_feat))
    if oversample and row["type"] == "classification":
        X_train, y_train = over_sample(X_train, y_train)
    #todo change this so it will save the data as a table inside the spark session
    # spark_df_joined = spark.createDataFrame(X_train)
    # spark_df_joined.createOrReplaceTempView("preprocess_results.X_train_{}".format(row["target"]))
    # spark_df_joined = spark.createDataFrame(y_train)
    # spark_df_joined.createOrReplaceTempView("preprocess_results.y_train_{}.csv".format(row["target"]))
    # spark_df_joined = spark.createDataFrame(X_train)
    # spark_df_joined.createOrReplaceTempView("preprocess_results.X_test_{}.csv".format(row["target"]))
    # spark_df_joined = spark.createDataFrame(X_train)
    # spark_df_joined.createOrReplaceTempView("preprocess_results.y_test_{}.csv".format(row["target"]))
    X_train.reset_index().to_csv("preprocess_results/X_train_{}.csv".format(row["target"]), index=False)
    y_train.to_csv("preprocess_results/y_train_{}.csv".format(row["target"]), index=False)
    X_test.reset_index().to_csv("preprocess_results/X_test_{}.csv".format(row["target"]), index=False)
    y_test.to_csv("preprocess_results/y_test_{}.csv".format(row["target"]), index=False)
    x = (row["target"], index, X_train, y_train.values, X_test, y_test.values, time_in_minutes, pipeline_feat)
    return x


def read_data(target, keys, spark):

    file_name = "preprocess_results/preprocess_pipeline_{}.p".format(target)
    target, index, time_in_minutes, pipeline_feat = load(open(file_name, "rb"))
    X_train = spark.sql("select * from preprocess_results.X_train_{}".format(target)).toPandas().set_index(keys)
    y_train = spark.sql("select * from preprocess_results.y_train_{}".format(target)).toPandas().values.flatten()
    X_test = spark.sql("select * from preprocess_results.X_test_{}".format(target)).toPandas().set_index(keys)
    y_test = spark.sql("select * from preprocess_results.y_test_{}".format(target)).toPandas().values.flatten()

    return target, index, X_train, y_train, X_test, y_test, time_in_minutes, pipeline_feat
