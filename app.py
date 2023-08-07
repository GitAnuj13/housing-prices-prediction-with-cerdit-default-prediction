
#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
plt.rcParams["figure.figsize"]=(20,10) # setting the default figure size

df=pd.read_csv("Bengaluru_House_Data.csv")
df.groupby("area_type")["area_type"].count()
df2=df.drop(["area_type","availability","society","balcony"],axis=1)
# filling na values with median
df2["bath"].fillna(df2["bath"].median(),inplace=True) 
df2.dropna(inplace=True)

df2["bhk"]=df2["size"].apply(lambda x: int(x.split(" ")[0]))
df3=df2.drop("size",axis=1)
def is_float(x):
    try:
        float(x)
        return True
    except:
        return False
    return True
def convert_sqrt(x):
    a=x.split("-")
    if len(a)==2:
        return (float(a[0])+float(a[1])/2)
    try:
        return float(x)
    except:
        return None
df4=df3.copy(deep=True)
df4["total_sqft"]=df4["total_sqft"].apply(convert_sqrt)
df4["location"]=df4["location"].apply(lambda x: x.strip())# removed all the trailing and leading whitespaces.
l_s=df4.groupby("location")["location"].count().sort_values(ascending=False) # Location constitutes more importance in recognizing the price
l_s_10=l_s[l_s<=10]
df4.location=df4.location.apply(lambda x: "Other_loc" if x in l_s_10 else x)

df4["price_sqft"]=df4["price"]*100000/df4["total_sqft"]
df5=df4[~(df4.total_sqft/df4.bhk<300)]
col=list(df5.columns)
        

location_group=df5.groupby("location")["location"].count().sort_values(ascending=False) # Location constitutes more importance in recognizing the price
location_group10=location_group[location_group<=10]
df5.location=df5.location.apply(lambda x: "Other_loc" if x in location_group10 else x)
def remove_outliers(df):
    df_1=pd.DataFrame()
    for key,df_2 in df.groupby('location'):
        mean=np.mean(df_2.price_sqft)
        std=np.std(df_2.price_sqft)
        df_3=df_2[(df_2.price_sqft>(mean-std))&(df_2.price_sqft<=(mean+std))]
        df_1=pd.concat([df_1,df_3],ignore_index=True)
    return df_1
df5=remove_outliers(df5)

def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_sqft),
                'std': np.std(bhk_df.price_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df7 = remove_bhk_outliers(df5)
# df8 = df7.copy()

plt.hist(df7.price)
plt.show()
df7=df7.drop("price_sqft",axis="columns")

dummies=pd.get_dummies(df7["location"],drop_first=True)
df7=pd.concat([df7,dummies.drop("Other_loc",axis="columns")],axis="columns")
x=df7.drop(["price","location"],axis="columns")
y=df7['price']
x.dropna(inplace=True)
y.dropna(inplace=True)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)
lr=LinearRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred))
from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor()
rf.fit(x_train,y_train)
y_pred_rf=rf.predict(x_test)
from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred_rf))
import streamlit as st
def predict(location,sqft,bath,bhk):
    print("Location:", location)
    print("Columns:", x.columns)  # Check if columns are present in the DataFrame x

    l_index = np.where(x.columns == location)[0][0]
    l_index=np.where(x.columns==location)[0][0]
    x1=np.zeros(len(x.columns))
    x1[0]=sqft
    x1[1]=bath
    x1[2]=bhk
    if l_index>=0:
        x1[l_index]=1
    return st.markdown("Price of the House is Rupees "+str(int(lr.predict([x1])[0]))+" Lakhs only")

selected_location = st.selectbox("Select an option for the location:", df7.location.unique())
selected_bathroom=st.selectbox("Select an option for the bathroom:", x["bath"].sort_values(ascending=True).unique())
selected_bhk=st.selectbox("Select an option for the number of bedroom:", x.bhk.sort_values(ascending=True).unique())
selected_sqft=st.selectbox("Select an option for the sqft area:", x.total_sqft.unique())
if st.button("House Price"):

    predict(selected_location,selected_sqft,selected_bathroom,selected_bhk)
st.title("Loan Eligiblity Prediction")
gender=st.text_input("Enter the Gender(0 for Feamle, 1 for Male)")    
credit_history=st.text_input("Enter the credit history(0 if credit score less than 750 , 1 if greter than 750)")
education=st.text_input("Enter the education (0 for graduate and 1 for non-graduate)")
loan_amount=st.text_input("Enter the Loan Amount(in Lakhs)")
applicant_income=st.text_input("Enter the applicant income(Monthly income in Lakhs)")
model = pickle.load(open('logistic_model.pkl','rb'))
input_data=([gender,credit_history,education,loan_amount,applicant_income])
if st.button("Loan eligiblity"):
    query=np.asarray(input_data,dtype=np.float64)
    reshaped_query=query.reshape(1,-1)
    output=int(abs(model.predict(reshaped_query)[0]))

    if output==1:
        st.title("Congratulations! Your Loan Got Accepted. ")
    else:
        st.markdown("Sorry! Your Loan Got Rejected.")
        st.markdown(" Try to apply for lower amount")



    

