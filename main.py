import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import time
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

st.write("""
# EV Charging Station Energy Usage App
This app makes predictions of the Energy Consumption at select EV Charging Stations in Scotland
""")
today = f'Todays Date is : {time.strftime("%Y-%m-%d")}'
st.write(today)
site = st.selectbox(
    'Select a Charging Site',
    ('Leslie','Rie_Achan','Broxden','Kinross','Friarton','South','Moness','Crown','King', 'Canal','Atholl','Mill')
)

if 'df' not in st.session_state:
    st.session_state['df'] = pd.DataFrame(columns=['Date', 'Temperature', 'Humidity', 'Precipitation', 'Windspeed'])

st.sidebar.header('User Input Parameters')

date = st.sidebar.date_input("Enter Prediction Start Date", datetime.date(2019, 8, 24), key='start_date')
temperature = st.sidebar.slider('Temperature (°C)', -8.0, 30.0, 10.0)
humidity = st.sidebar.slider('Humidity (%)', 50.0, 100.0, 80.0)
precipitation = st.sidebar.slider('Precipitation (mm)', 0.00, 70.0, 50.0)
windspeed = st.sidebar.slider('Windspeed (kph)', 5.0, 70.0, 20.0)
def user_input_features():
    data = {'date': [date],
            'temp': [temperature],
            'humidity': [humidity],
            'precip': [precipitation],
            'windspeed': [windspeed]}
    features = pd.DataFrame(data)
    return features

df = user_input_features()



data = pd.read_csv(f"site data/{site}.csv",index_col=0)
plt.figure(figsize=(10,5))
plt.plot(data.index, data['daily'])
plt.title(f'Historical Energy Consumption for {site} Site')
plt.ylabel('Energy Consumption(KWh)')
plt.xlabel('Date')
if not isinstance(data.index, pd.DatetimeIndex):
    data.index = pd.to_datetime(data.index)
# Identify the positions (integers) for x-ticks
tick_positions = range(0, len(data.index), int(len(data.index)/10))  # Adjust this as needed
# Convert the selected tick positions to formatted strings
tick_labels = [data.index[i].strftime('%Y-%m-%d') for i in tick_positions]
# Set the new ticks and labels
plt.xticks(ticks=tick_positions, labels=tick_labels)
plt.xticks(rotation=45)
st.pyplot(plt)

if st.sidebar.button('Add Values'):
    new_data= user_input_features()
    if st.session_state['df'].empty:
        st.session_state['df'] = new_data
    else:
        st.session_state['df'] = pd.concat([st.session_state['df'], new_data], ignore_index=True)
    st.subheader('User Input Parameters')
    st.write(st.session_state['df'])

data = data.asfreq('D')
data = data.interpolate()
for i in range(1,4):
    data[f'lag_{i}'] = data['daily'].shift(i)

data = data.dropna()
split_point = int(len(data) * 0.8)
train = data.iloc[:split_point]
test = data.iloc[split_point:]

X_train = train.drop('daily', axis = 1)
y_train = train['daily']
y_test = test['daily']
xgb_model = XGBRegressor(booster='gbtree', n_estimators = 500, max_depth = 3, learning_rate = 0.01)
xgb_model.fit(X_train, y_train)
X_test = test.drop('daily', axis=1)
predictions = xgb_model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
maes = f'The Mean Absolute Error is: {str(mae)[:6]} KWh'

# if 'df' in st.session_state:
#     all_data = st.session_state['df']
#     # all_data_df = pd.DataFrame(all_data)
# st.write(all_data)
# input_data = all_data.set_index('date')
# st.write(input_data)
# for i in range(1,4):
#     input_data[f'lag_{i}'] = input_data['temp'].shift(i)
# input_data = input_data.dropna()
# st.write(input_data)
# predicted_energy = xgb_model.predict(input_data)
# preds = pd.DataFrame({
#     'Date': input_data.index,
#     'Consumption': predicted_energy
# })
# st.write(preds)
if st.button('Make Predictions'):
    if 'df' in st.session_state:
        all_data = st.session_state['df']
        input_data = all_data.set_index('date')

        # Creating lags
        for i in range(1, 4):
            input_data[f'lag_{i}'] = input_data['temp'].shift(i)

        # Dropping rows with NaN values that were created due to lagging
        input_data = input_data.dropna()

        # Making predictions
        predicted_energy = xgb_model.predict(input_data)

        # Preparing the predictions DataFrame
        preds = pd.DataFrame({
            'Date': input_data.index,
            'Consumption': predicted_energy
        })

        # Displaying the predictions DataFrame
        st.subheader('User Input Parameters')
        st.write(st.session_state['df'])
        st.subheader('Predictions')
        st.write(preds)
        st.write(maes)