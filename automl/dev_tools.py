import pandas as pd
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from tensorflow.compat.v1.keras import backend as K
import json
import io
import joblib


def recall_m(y_true, y_pred):
   true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
   possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
   recall = true_positives / (possible_positives + K.epsilon())
   return recall


def precision_m(y_true, y_pred):
   true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
   predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
   precision = true_positives / (predicted_positives + K.epsilon())
   return precision


def f1_m(y_true, y_pred):
   precision = precision_m(y_true, y_pred)
   recall = recall_m(y_true, y_pred)
   return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def is_date(df, col, string_ratio=0.02):
    """
    check if a column in a dataframe is a date or not
    :param df: the dataframe to check if it's ok - Dataframe
    :param col: the column to operate over - string
    :param string_ratio: the ratio that deside if there failed attempts / size of vector percentage larger than ratio
    than the feature is not a date - float
    :return: if the column is a date or not - boolearn
    """
    count = 1
    try:
        value_list = df[col].fillna(0).unique().tolist()
    except Exception as e:
        print(df[col])
        print(e)
        print(col)

    for value in value_list:
        try:
            pd.Timestamp(value)
        except Exception as e:
            count += 1
            if count / len(value_list) >= string_ratio:
                return False

    return True


def get_cols(df, exclude=[], ratio=0.1, key_cols=[]):
    """
    finds the different types of features automatically
    :param df: the dataframe to check for columns - Dataframe
    :param exclude: columns to ignore - list of strings
    :param ratio: this ratio decide if a numeric column is actually categoric if the number of unique values in the
    feature divided by the size of feature is less than the ratio - float
    :return: dictionary with keys as columns types and values that are lists of strings with column names - dictionary
    """
    key_cols = [col for col in df.columns.tolist() if "_id" in col.lower()] + key_cols
    cat_cols = df.select_dtypes(include=["object", "bool"]).columns.tolist()
    date_cols = [col for col in cat_cols if col not in exclude + key_cols and is_date(df, col)]
    cat_cols = [col for col in cat_cols if col not in date_cols + key_cols + exclude]
    numeric_cols = [col for col in df.columns.tolist() if col not in key_cols + date_cols + cat_cols + exclude]

    for col in numeric_cols:
        try:
            if df[col].unique().shape[0] <= ratio * df.shape[0]:
                cat_cols.append(col)
        except Exception as e:
            print(e)

    numeric_cols = [col for col in numeric_cols if col not in cat_cols]

    return {"key": key_cols, "categoric": cat_cols, "date": date_cols, "numeric": numeric_cols}


def string_2json(x):

    try:
        if isinstance(x, float):
            print("nan")
            x = {}
            return x
        try:
            x
        except Exception as e:
            x = {}
            return x
        clean = x.replace("':", '":').replace(", '", ', "').replace("{'", '{"').replace("'}", '"}').replace("array(", "").replace("])", "]").replace(", dtype=int64)", "").replace(".        ", "").replace('\\n       ', "").replace('. ', "").replace(".,", ".0,").replace(".]", ".0]").replace("...", "").replace("dtype=float32),", "").replace("nan", "0")
        y = json.loads(clean)
        x = y
        return x
    except Exception as e:
        print(e)


def save(bytes_container: io.BytesIO, o):
    """

    :param bytes_container:
    :return:
    """
    #dump model
    byte_io = io.BytesIO()
    joblib.dump(o, byte_io)
    pack = {'class_dict':  {'a': 12231},
            'model': byte_io.getvalue()}
    joblib.dump(pack, bytes_container)


def load(bytes_container: io.BytesIO):
    """

    :param bytes_container:
    :return:
    """
    pack = joblib.load(bytes_container)
    o = joblib.load(io.BytesIO(pack["model"]))
    return o


def send_email(params, mail_content="ok", subject='test mail'):

    #The mail addresses and password
    sender_address = params["email"]
    sender_pass = params["password"]
    receiver_addresses = params["recipients"]
    #Setup the MIME
    message = MIMEMultipart()
    message['From'] = sender_address
    for to in receiver_addresses:
        message['To'] = to
        message['Subject'] = subject   #The subject line
        #The body and the attachments for the mail
        message.attach(MIMEText(mail_content, 'plain'))
        #Create SMTP session for sending the mail
        session = smtplib.SMTP('smtp.gmail.com', 587) #use gmail with port
        session.starttls() #enable security
        session.login(sender_address, sender_pass) #login with mail_id and password
        text = message.as_string()
        session.sendmail(sender_address, to, text)
        session.quit()
        print('Mail Sent to: {}'.format(to))
