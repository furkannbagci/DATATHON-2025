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
df_train.info()
df_train.nunique()


#********************************************************************************************

#EVENT_TİME DATETİME ÇEVİRME VE AYRIŞTIRMA
df_train["event_time"] = pd.to_datetime(df_train["event_time"])
df_test['event_time'] = pd.to_datetime(df_test["event_time"])

def tarih_ayristirma(df,column,görsel_true):
    df["year"] = df[column].dt.year
    df["month"] = df[column].dt.month
    df["day"] = df[column].dt.day
    df["hour"] = df[column].dt.hour
    df["weekday"] = df[column].dt.weekday  # 0: Pazartesi, 6: Pazar

    degerler = ["year", "month", "day", "hour", "weekday"]
    for col in degerler:
        print(col," : ",df[col].nunique())
    if görsel_true==1:
        for col in degerler:
            plt.figure(figsize=(6, 4))
            sns.countplot(x=col, data=df)
            plt.title(f"{col} Bazlı Dağılım")
            plt.show()
tarih_ayristirma(df_train,"event_time",0)
tarih_ayristirma(df_test,"event_time",0)

#gün döngüsü yakalama
def add_time_features(df):
    df["hour"] = df["event_time"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    return df

df_train = add_time_features(df_train)
df_test = add_time_features(df_test)

#********************************************************************************************
tekrar_eden_train = df_train.groupby("user_id").agg(
    user_session =("user_session", list),
    count=("user_session", "count")
)

tekrarlılar = tekrar_eden_train[tekrar_eden_train["count"] > 1]

print(tekrarlılar)
##********************************************************************************************
"""
df_train_sorted = df_train.sort_values(by=['user_session', 'event_time'])
df_train_sorted['time_diff'] = df_train_sorted.groupby('user_session')['event_time'].diff().dt.total_seconds()
time_diff_features = df_train_sorted.groupby('user_session')['time_diff'].agg(['mean', 'std']).fillna(0)

df_test_sorted = df_test.sort_values(by=['user_session', 'event_time'])
df_test_sorted['time_diff'] = df_test_sorted.groupby('user_session')['event_time'].diff().dt.total_seconds()
time_diff_features_test = df_test_sorted.groupby('user_session')['time_diff'].agg(['mean', 'std']).fillna(0)
"""

user_event_counts = df_train.groupby(['user_id', 'event_type']).size().unstack(fill_value=0)
user_event_counts['purchase_rate'] = user_event_counts.get('BUY', 0) / (user_event_counts.sum(axis=1) + 1e-6)

user_event_counts_test = df_test.groupby(['user_id', 'event_type']).size().unstack(fill_value=0)
user_event_counts_test['purchase_rate'] = user_event_counts_test.get('BUY', 0) / (user_event_counts_test.sum(axis=1) + 1e-6)


product_popularity = df_train['product_id'].value_counts().to_frame('product_popularity')
df_train_with_pop = df_train.join(product_popularity, on='product_id')

product_popularity_test = df_test['product_id'].value_counts().to_frame('product_popularity')
df_test_with_pop = df_test.join(product_popularity, on='product_id')



#********************************************************************************************

#SESSİON_VALUE ÖLÇÜMLERİ
weekday_ortalamaları = df_train.groupby("weekday")["session_value"].mean().reset_index()
event_type_ortalamalar = df_train.groupby("event_type")["session_value"].mean().reset_index()
gun_saati = df_train.groupby("hour")["session_value"].mean().reset_index()

category_id = df_train.groupby("category_id")["session_value"].mean().reset_index()
category_id = category_id[category_id['session_value'] > category_id['session_value'].mean()]

cat_event = df_train.groupby(["event_type","category_id"])["session_value"].mean().reset_index()
cat_event = cat_event[cat_event['session_value'] > cat_event['session_value'].mean()]

pivot = cat_event.pivot(index="event_type",columns = "category_id",values = "session_value")
sns.heatmap(pivot, cmap="coolwarm")


plt.bar(category_id["product_id"], category_id["session_value"])
plt.show()

plt.bar(weekday_ortalamaları["weekday"],weekday_ortalamaları["session_value"])
plt.xlabel("Weekday")
plt.ylabel("Session Value")
plt.show()

plt.bar(event_type_ortalamalar["event_type"],event_type_ortalamalar["session_value"])
plt.xlabel("Event Type")
plt.ylabel("Session Value")
plt.show()

plt.bar(gun_saati["hour"],gun_saati["session_value"])
plt.xlabel("hour")
plt.ylabel("Session Value")
plt.show()
#pivot tablo

pivot = df_train.pivot_table(values="session_value", index="weekday", columns="event_type", aggfunc="mean")
plt.imshow(pivot,cmap="Blues",aspect="auto")
plt.colorbar(label="Session Value")
plt.xticks(range(len(pivot.columns)),pivot.columns)
plt.yticks(range(len(pivot.index)),pivot.index)
plt.show()

#outlier tespiti
df_train["session_value"].plot(kind="box")
plt.title("session_value")
plt.show()

#********************************************************************************************

#ortak user_id değerlerini bulma
def ortak_degerler(df_train,df_test,col):
    ortak =set(df_train[col]) & set(df_test[col])
    tr_ortak = df_train[df_train[col].isin(ortak)]
    te_ortak = df_test[df_test[col].isin(ortak)]

    print(f"ortak {col} sayisi : ", len(ortak))
    print(f"tr_ortak {col} sayisi : ", len(tr_ortak))
    print(f"te_ortak {col} sayisi : ", len(te_ortak))
ortak_degerler(df_train,df_test,"user_id")
#********************************************************************************************


#SESSİON BAZLI AYRIŞIM YAPMAK GEREKİYOR
session_features = df_train.groupby("user_session").agg(
    unique_category = ("category_id", "nunique"),
    session_start = ("event_time", "min"),
    session_end = ("event_time", "max"),
    weekday = ("weekday","mean"),
    hour_sin = ("hour_sin", "mean"),
    hour_cos = ("hour_cos", "mean"),
)

session_features_test = df_test.groupby("user_session").agg(
    unique_category = ("category_id", "nunique"),
    session_start = ("event_time", "min"),
    session_end = ("event_time", "max"),
    weekday = ("weekday","mean"),
    hour_sin=("hour_sin", "mean"),
    hour_cos=("hour_cos", "mean"),
)
"""
#islem süre farkı
session_features["islem_suresi"] = (
    (session_features["session_end"] - session_features["session_start"]).dt.total_seconds())

session_features_test["islem_suresi"] = (
    (session_features_test["session_end"] - session_features_test["session_start"]).dt.total_seconds()
)
"""

#her bir user_session için event_type tablosu
event_type_counts = pd.crosstab(df_train["user_session"], df_train["event_type"])
event_type_counts_test = pd.crosstab(df_test["user_session"], df_test["event_type"])

#y değeri
session_targets = df_train.groupby("user_session")["session_value"].mean()

#kullanıcı bazlı
user_profiles = df_train.groupby("user_id").agg(
    # Aktivite Hacmi
    user_total_sessions=('user_session', 'nunique'),
    user_total_events=('event_type', 'count'),
    # Davranışsal Event Sayıları
    user_total_views=('event_type',lambda x: (x == 'VIEW').sum()),
    user_total_add_carts=('event_type',lambda x: (x == 'ADD_CART').sum()),
    user_total_buys=('event_type',lambda x: (x == 'BUY').sum()),
    # Etkileşim Çeşitliliği
    user_unique_products_interacted=('product_id', 'nunique')
).reset_index()

# Davranışsal Oranları Hesaplama (Kullanıcının satın alma eğilimi vs.)
epsilon = 1e-6
user_profiles['user_buy_to_view_ratio'] = user_profiles['user_total_buys'] / (user_profiles['user_total_views'] + epsilon)
user_profiles['user_add_cart_to_view_ratio'] = user_profiles['user_total_add_carts'] / (user_profiles['user_total_views'] + epsilon)


user_profiles_test = df_test.groupby("user_id").agg(
    # Aktivite Hacmi
    user_total_sessions=('user_session', 'nunique'),
    user_total_events=('event_type', 'count'),
    # Davranışsal Event Sayıları
    user_total_views=('event_type',lambda x: (x == 'VIEW').sum()),
    user_total_add_carts=('event_type',lambda x: (x == 'ADD_CART').sum()),
    user_total_buys=('event_type',lambda x: (x == 'BUY').sum()),
    # Etkileşim Çeşitliliği
    user_unique_products_interacted=('product_id', 'nunique')
).reset_index()

# Davranışsal Oranları Hesaplama (Kullanıcının satın alma eğilimi vs.)
user_profiles_test['user_buy_to_view_ratio'] = user_profiles_test['user_total_buys'] / (user_profiles_test['user_total_views'] + epsilon)
user_profiles_test['user_add_cart_to_view_ratio'] = user_profiles_test['user_total_add_carts'] / (user_profiles_test['user_total_views'] + epsilon)

#********************************************************************************************

#train
session_user = df_train.groupby("user_session")["user_id"].first().to_frame()
session_features = session_features.join(session_user, on="user_session")
session_features = session_features.join(user_profiles, on="user_id")

#test
session_user_test = df_test.groupby("user_session")["user_id"].first().to_frame()
session_features_test = session_features_test.join(session_user_test, on="user_session")
session_features_test = session_features_test.join(user_profiles_test, on="user_id")


session_features = session_features.join(user_event_counts[['purchase_rate']], on='user_id')
#session_features = session_features.join(time_diff_features)
session_product_pop = df_train_with_pop.groupby('user_session')['product_popularity'].mean().to_frame('avg_product_popularity')
session_features = session_features.join(session_product_pop)

session_features_test = session_features_test.join(user_event_counts_test[['purchase_rate']], on='user_id')
#session_features_test = session_features_test.join(time_diff_features_test)
session_product_pop_test = df_test_with_pop.groupby('user_session')['product_popularity'].mean().to_frame('avg_product_popularity')
session_features_test = session_features_test.join(session_product_pop_test)


final_features = session_features.join(event_type_counts)
final_features_test = session_features_test.join(event_type_counts_test)

#********************************************************************************************

# 0'a bölme hatasını engellemek için küçük bir sayı (epsilon) ekliyoruz
epsilon = 1e-6

# Görüntülemeden sepete ekleme oranı
final_features['view_to_add_cart_rate'] = event_type_counts['ADD_CART'] / (event_type_counts['VIEW'] + epsilon)

# Sepete eklemeden satın alma oranı
final_features['add_cart_to_buy_rate'] = event_type_counts['BUY'] / (event_type_counts['ADD_CART'] + epsilon)

# Test seti için de aynısını yapmayı unutma!
final_features_test['view_to_add_cart_rate'] = event_type_counts_test['ADD_CART'] / (event_type_counts_test['VIEW'] + epsilon)
final_features_test['add_cart_to_buy_rate'] = event_type_counts_test['BUY'] / (event_type_counts_test['ADD_CART'] + epsilon)

product_agg = df_train.groupby('product_id').agg(
    product_views=('event_type', lambda x: (x == 'VIEW').sum()),
    product_buys=('event_type', lambda x: (x == 'BUY').sum())
).reset_index()

product_agg_test = df_test.groupby('product_id').agg(
    product_views=('event_type', lambda x: (x == 'VIEW').sum()),
    product_buys=('event_type', lambda x: (x == 'BUY').sum())
).reset_index()

# Ürünün satın alma gücü
product_agg['product_conv_rate'] = product_agg['product_buys'] / (product_agg['product_views'] + epsilon)
product_agg_test['product_conv_rate'] = product_agg_test['product_buys'] / (product_agg_test['product_views'] + epsilon)

# Bu bilgiyi ana event tablosuna ekle
df_train_merged = pd.merge(df_train, product_agg[['product_id', 'product_conv_rate']], on='product_id', how='left').fillna(0)
df_test_merged = pd.merge(df_test,product_agg[['product_id', 'product_conv_rate']],on='product_id', how='left').fillna(0)

# Her oturumdaki ürünlerin ortalama dönüşüm oranını hesapla
session_avg_product_conv = df_train_merged.groupby('user_session')['product_conv_rate'].mean().to_frame('session_avg_product_conv')
session_avg_product_conv_test = df_test_merged.groupby("user_session")['product_conv_rate'].mean().to_frame('session_avg_product_conv')

# Son olarak ana özellik tablomuza ekleyelim
final_features = final_features.join(session_avg_product_conv)
final_features_test = final_features_test.join(session_avg_product_conv_test)


#********************************************************************************************
#son hali temizlenmiş
final_features = final_features.drop(columns=["session_start", "session_end","user_id"])  # ham datetime'ları atabiliriz
final_features_test = final_features_test.drop(columns=["session_start", "session_end","user_id"])
#********************************************************************************************


train = final_features
y = session_targets

test = final_features_test

copy_train_df = train.copy()
#session_value log işlemi outlier
copy_train_df["session_value"] = y
copy_train_df["session_value"] = np.log1p(copy_train_df["session_value"])

#********************************************************************************************
import statsmodels.api as sm
y = copy_train_df["session_value"]  # Bağımlı değişkenini buraya ekle
X = copy_train_df.iloc[:,:-1]  # Bağımsız değişkenler
# X matrisine sabit terim ekle
X = sm.add_constant(X)

# Modeli oluştur ve fit et
model = sm.OLS(y, X).fit()
print(model.summary())

corr_matrix=copy_train_df.corr()

sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title("Correlation Heatmap")
plt.show()

from statsmodels.stats.outliers_influence import variance_inflation_factor
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# VIF yüksek olanları sırala
vif_data = vif_data.sort_values(by="VIF", ascending=False)
print(vif_data)

#********************************************************************************************
from autogluon.tabular import TabularPredictor

TRAIN = True
MODEL_PATH = "./autogluon_datathon_model"

if TRAIN:
    predictor = TabularPredictor(
        label="session_value",
        problem_type="regression",
        eval_metric="mean_squared_error",
        path=MODEL_PATH
    ).fit(copy_train_df,presets="best_quality",time_limit=3600)
else:
    predictor = TabularPredictor.load(MODEL_PATH)

leaderboard = predictor.leaderboard(copy_train_df, silent=True)

predictor.feature_importance(data=copy_train_df)
copy_train_df.info()
print(leaderboard)

#-------------------------------------------------------------------------------

y_pred = predictor.predict(test)
final_predictions = np.expm1(y_pred)

submission =pd.DataFrame({
    "user_session" : final_features_test.index,
    "session_value" : final_predictions
})

submission.to_csv("submission.csv", index=False)


