import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st

st.title('Stock Trend Prediction')

user_input= st.text_input("Enter Stock Ticker", 'TSLA')

start ='2017-05-14'
end  ='2019-05-12'
df=data.get_data_tiingo(user_input,start,end,api_key='e064da3d7204d772e0365bc82d9881329c0bdf46')


#DESCRIBING DATA
st.subheader('Data from 2015-2019')
st.write(df.describe())

#VISUALISATIONS

st.subheader('Closing Price Vs Time Chart')
fig=plt.figure(figsize=(12,6))
plt.plot(df['close'].to_numpy(), label='Close')

st.pyplot(fig)


st.subheader('Closing Price Vs Time Chart With 100MA')
ma100=df['close'].rolling(100).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100.to_numpy(), label='Close')
plt.plot(df['close'].to_numpy(), label='Close')
st.pyplot(fig)

st.subheader('Closing Price Vs Time Chart With 100MA & 200MA')
ma100=df['close'].rolling(100).mean()
ma200=df['close'].rolling(200).mean()

fig=plt.figure(figsize=(12,6))
plt.plot(ma100.to_numpy(),'r', label='Close')
plt.plot(ma200.to_numpy(),'g', label='Close')
plt.plot(df['close'].to_numpy(),'b', label='Close')
st.pyplot(fig)







data_training=pd.DataFrame(df['close'][0:int(len(df)*0.70)])
data_testing=pd.DataFrame(df['close'][int(len(df)*0.70):int(len(df))])


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
data_training_array=scaler.fit_transform(data_training)



#LOAD MY MODEL

model = load_model('keras_model.h5')

#Testing Part

past_100_days=data_training.tail(100)
final_df=past_100_days.append(data_testing,ignore_index=True)
input_data=scaler.fit_transform(final_df)


x_test=[]
y_test=[]

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])
    
x_test , y_test=  np.array(x_test),np.array(y_test)

y_predicted =model.predict(x_test)
scaler.scale_
scale_factor=1/scaler.scale_[0]
y_predicted=y_predicted* scale_factor
y_test = y_test* scale_factor

#FINAL GRAPH
st.subheader('Predicted Vs Orignal')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label="Original Price")
plt.plot(y_predicted,'r',label="Predicted Price")
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)