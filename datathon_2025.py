import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

pd.set_option("display.max_columns",None)
pd.set_option("display.width",500)

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

#VERİLERİ KONTROL ETME
df_train.dtypes
df_train.nunique()
for col in df_train.columns:
    print(col ,"null değer sayısı : ",df_train[col].isnull().sum())
    print(col ,"nan değer sayısı",df_train[col].isna().sum())


#ENCODER İLE OBJECT VERİLERİ SAYISAL HALE GETİRME
columns = ["product_id","category_id","user_id","user_session","event_type"]
encoder = LabelEncoder()
for col in columns:
    df_train[col] = encoder.fit_transform(df_train[col])



#EVENT_TİME DATETİME ÇEVİRME VE AYRIŞTIRMA
df_train["event_time"] = pd.to_datetime(df_train["event_time"])
df_test['event_time'] = pd.to_datetime(df_test["event_time"])

def tarih_ayristirma(df,column):
    df["year"] = df[column].dt.year
    df["month"] = df[column].dt.month
    df["day"] = df[column].dt.day
    df["hour"] = df[column].dt.hour
    df["weekday"] = df[column].dt.weekday  # 0: Pazartesi, 6: Pazar

    degerler = ["year", "month", "day", "hour", "weekday"]
    for col in degerler:
        print(col," : ",df[col].nunique())

    for col in degerler:
        plt.figure(figsize=(6, 4))
        sns.countplot(x=col, data=df)
        plt.title(f"{col} Bazlı Dağılım")
        plt.show()
tarih_ayristirma(df_train,"event_time")
tarih_ayristirma(df_test,"event_time")


tekrar_eden = df_train.groupby("user_id").agg(
    user_session =("user_session", list),
    count=("user_session", "count")
)

duplicate = tekrar_eden[tekrar_eden["count"] > 1]

print(duplicate)

#-----------------------------------------------------------------

#SESSİON BAZLI AYRIŞIM YAPMAK GEREKİYOR
session_features = df_train.groupby("user_session").agg(
    total_event = ("event_type", "sum"),
    unique_event = ("event_type", "nunique"),
    unique_products = ("product_id", "nunique"),
    unique_category = ("product_id", "nunique"),
    session_start = ("event_time", "min"),
    session_end = ("event_time", "max")
)

#islem süre farkı
session_features["islem_suresi"] = (
    (session_features["session_end"] - session_features["session_start"]).dt.total_seconds())

#her bir user_session için event_type tablosu
event_type_counts = pd.crosstab(df_train["user_session"], df_train["event_type"])

#kullanıcı bazlı
user_stats = df_train.groupby("user_id").agg(
    user_total_sessions=("user_session", "nunique"),
    user_total_events=("event_type", "count"),
    user_avg_value=("session_value", "mean"),
    user_total_value=("session_value", "sum")
)


session_user = df_train.groupby("user_session")["user_id"].first().to_frame()
session_features = session_features.join(session_user, on="user_session")
session_features = session_features.join(user_stats, on="user_id")

session_targets = df_train.groupby("user_session")["session_value"].sum()

final_features = session_features.join(event_type_counts)
final_features = final_features.drop(columns=["session_start", "session_end"])  # ham datetime'ları atabiliriz

final_features.columns = final_features.columns.astype(str)

x = final_features
y = session_targets

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100)
rf.fit(x,y)
y_pred = rf.predict(x)
print(y_pred[:20])

#outlier tespiti
df_train["session_value"].plot(kind="box")
plt.title("session_value")
plt.show()


