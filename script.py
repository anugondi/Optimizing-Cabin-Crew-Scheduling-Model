# %%
import numpy as py
import matplotlib.pyplot as plt
import seaborn as so
import xgboost
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from xgboost import plot_importance
from sklearn.metrics import r2_score
import streamlit as st

# %%
import pandas as pd

df_unclean = pd.read_excel('Business_Dataset.xlsx')

#print(df_unclean.size)
#print(df_unclean.head)

# %%
df = df_unclean.drop(columns=["Geocoded_City1", "Geocoded_City2"])
df = df.dropna()

df['fare_low'] = pd.to_numeric(df['fare_low'], errors='coerce')
df['fare_lg'] = pd.to_numeric(df['fare_lg'], errors='coerce')
df['passengers'] = py.where(df['passengers'] >= 800, 800, df['passengers'])

#print(df.head)
#print(df.size)

#print(df.columns)

# %% [markdown]
# Exploratory Analaysis
# 1) Average miles per flight per per year (1993 - 2024)
# 2) Average passenger per flight per year (1993 - 2024)
st.title("Optimized Crew Scheduling")


# %%
# Average miles and passengers per year (1993 - 2024)

grouped_df = df.groupby('Year').agg({
    'nsmiles': 'mean',
    'passengers': 'mean'
}).reset_index()

print(grouped_df)

# Insights reveal:
# Passengers- Increasing number of passengers per flight from 1993 to 2024
# Miles- Average miles per flight per year is relatively constant, indicating how flight occurences have increased with passenger travel frequency. 

# %%
plt.figure(figsize=(10, 6))
so.barplot(data=grouped_df, x='Year', y='passengers', palette='viridis')

plt.title('Average Number of Passengers per Flight per Year')
plt.xlabel('Year')
plt.ylabel('Average Number of Passengers per Flight')
plt.xticks(rotation=45, ha='right')
plt.show()

# %%

plt.figure(figsize=(10, 6))
so.barplot(data=grouped_df, x='Year', y='nsmiles', palette='viridis')

plt.title('Average Number of Miles per Flight per Year')
plt.xlabel('Year')
plt.ylabel('Average Number of Miles per Flight')
plt.xticks(rotation=45, ha='right')
plt.show()


# %%
plt.figure(figsize=(10, 10))
so.scatterplot(data=df, x='passengers', y='fare_low', hue='nsmiles', size="nsmiles")

plt.title('Number of Passengers vs Low Fare')
plt.xlabel('Fare ($)')
plt.ylabel('Passengers')
plt.show()




# %%
plt.figure(figsize=(10, 10))
so.scatterplot(data=df, x='passengers', y='fare_lg', hue='nsmiles', size="nsmiles")

plt.title('Number of Passengers vs High Fare')
plt.xlabel('Fare ($)')
plt.ylabel('Passengers')
plt.show()


# %%
plt.figure(figsize=(10, 6))
carrier_df = df.groupby(['carrier_low'])['carrier_low'].count().reset_index(name = "count").sort_values(by = 'count', ascending = False)

plt.figure(figsize=(30, 6))
so.barplot(data=carrier_df, x='carrier_low', y='count', palette='viridis')

# Add titles and labels
plt.title('Number of Flights per Carrier Low')
plt.xlabel('Carrier')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.show()

# carrier_df.head


# %%
F9_df = df[df["carrier_low"] == "F9"]
F9_df.shape

# %%
X = F9_df[['fare_low']]
Y = F9_df[['passengers']]

x_train,x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=6)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
r2 = r2_score(y_test, y_pred)
print(f"R-squared: {r2}")

plt.figure(figsize=(10, 7))
plot_importance(model)
plt.show()



# %%
X = F9_df[['nsmiles']]

x_train,x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=6)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
r2 = r2_score(y_test, y_pred)
print(f"R-squared: {r2}")

plt.figure(figsize=(10, 7))
plot_importance(model)
plt.show()



# %%
X = F9_df[['quarter']]

x_train,x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=6)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
r2 = r2_score(y_test, y_pred)
print(f"R-squared: {r2}")

plt.figure(figsize=(10, 7))
plot_importance(model)
plt.show()


# %%
X = F9_df[['nsmiles', 'fare_low']]

x_train,x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=6)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
r2 = r2_score(y_test, y_pred)
print(f"R-squared: {r2}")


plt.figure(figsize=(10, 7))
plot_importance(model)
plt.show()


# %%
X = F9_df[['nsmiles', 'fare_low', 'quarter']]

x_train,x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=6)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
r2 = r2_score(y_test, y_pred)
print(f"R-squared: {r2}")


plt.figure(figsize=(10, 7))
plot_importance(model)
plt.show()


# %%
X = F9_df[['nsmiles', 'fare_low']]

x_train,x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=6)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
r2 = r2_score(y_test, y_pred)
print(f"R-squared: {r2}")

plt.figure(figsize=(10, 7))
plot_importance(model)
plt.show()


# %%
def predict_passengers(fare, nsmiles):
    input_data = py.array([[nsmiles, fare]])  
    prediction = model.predict(input_data)  
    return prediction[0]  

fare_input = 200  
nsmiles_input = 1000  

predicted_passengers = predict_passengers(fare_input, nsmiles_input)
print(f"Predicted number of passengers: {predicted_passengers}")

# %%

plt.figure(figsize=(10, 6))

so.scatterplot(x=y_test.values.flatten(), y=y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()])
plt.xlabel("Actual Passengers")
plt.ylabel("Predicted Passengers")
plt.title("Actual vs Predicted Passengers")
plt.show()

residuals = y_test.values.flatten() - y_pred
plt.figure(figsize=(10, 6))
so.histplot(residuals, kde=True)
plt.title("Residuals Distribution")
plt.show()


# %%

plt.figure(figsize=(10, 6))
bins = [0, 500, 1000, 1500, 2000, 2500, 2724]
labels = ['0-500', '500-1000', '1000-1500', '1500-2000', '2000-2500', '2500-2724']

F9_df['nsmiles_binned'] = pd.cut(F9_df['nsmiles'], bins=bins, labels=labels, include_lowest=True)

print(F9_df[['nsmiles', 'nsmiles_binned']].head())

so.boxplot(x='nsmiles_binned', y='passengers', data = F9_df)
plt.title('Box Plot of Passengers vs Number of Miles')
plt.xlabel('Number of Miles')
plt.ylabel('Number of Passengers')

grouped = F9_df.groupby('nsmiles_binned')['passengers']
for i, group in enumerate(grouped):
    first = group[1].quantile(0.25)  
    median = group[1].quantile(0.5)
    third = group[1].quantile(0.75)  
    iqr = third - first  
    print(f"Bin {group[0]}: Q1 = {first}, Median = {median}, Q3 = {third}, IQR = {iqr}")

plt.show()


# %%
fare_input = 150 
nsmiles_input = 1200  

predicted_passengers = predict_passengers(fare_input, nsmiles_input)
print(f"Predicted number of passengers: {predicted_passengers}")

capacity_F9  = {
    500 : 298,
    1000 : 541,
    1500 : 264,
    2000 : 274,
    2500 : 140,
    2724 : 11
    }

bucket = 0
for key in capacity_F9:
    if nsmiles_input <= key:
        bucket = key
        break
if predicted_passengers > capacity_F9.get(bucket):
    predicted_passengers = capacity_F9.get(bucket)
    
print(f"Predicted number of passengers: {predicted_passengers}")



# %%
NK_df = df[df["carrier_low"] == "NK"]
NK_df.shape

X = NK_df[['nsmiles', 'fare_low']]
Y = NK_df[['passengers']]

x_train,x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=6)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
r2 = r2_score(y_test, y_pred)
print(f"R-squared: {r2}")

plt.figure(figsize=(10, 7))
plot_importance(model)
plt.show()


# %%
#spirit
plt.figure(figsize=(10, 6))
bins = [0, 500, 1000, 1500, 2000, 2500, 2724]
labels = ['0-500', '500-1000', '1000-1500', '1500-2000', '2000-2500', '2500-2724']

NK_df['nsmiles_binned'] = pd.cut(NK_df['nsmiles'], bins=bins, labels=labels, include_lowest=True)

print(NK_df[['nsmiles', 'nsmiles_binned']].head())

so.boxplot(x='nsmiles_binned', y='passengers', data = NK_df)
plt.title('Box Plot of Passengers vs Number of Miles')
plt.xlabel('Number of Miles')
plt.ylabel('Number of Passengers')

plt.show()

def output(input_fare, input_miles):
    predicted_passengers = predict_passengers(input_fare, input_miles)
    print(f"Predicted number of passengers: {predicted_passengers}")

    bucket = 0

    capacity_F9  = {
    500 : 298,
    1000 : 541,
    1500 : 264,
    2000 : 274,
    2500 : 140,
    2724 : 11
    }
    
    for key in capacity_F9:
        if nsmiles_input <= key:
            bucket = key
            break
    if predicted_passengers > capacity_F9.get(bucket):
        predicted_passengers = capacity_F9.get(bucket)
        
    print(f"Predicted number of passengers: {predicted_passengers}")

    crew  = {
        18 : 0,
        50 : 1,
        100 : 2
    }
    attendants = -1

    for key in crew:
        if predicted_passengers <= key:
            attendants = crew.get(key)
            break
        
    if attendants == -1:
        predicted_passengers -= 100
        attendants = predicted_passengers // 50 + 3
    print(f"Predicted number of attendants: {attendants}") 

    return [predicted_passengers, attendants]

# %%
st.subheader("Input Fare and Number of Miles for Crew Prediction - Frontier Airlines")

if 'fare' not in st.session_state:
    st.session_state.fare = 0.0
if 'nmiles' not in st.session_state:
    st.session_state.nmiles = 0.0

st.session_state.fare = st.number_input("Fare ($):", min_value=0.0, step=1.0, value=st.session_state.fare)
st.session_state.nmiles = st.number_input("Number of Miles:", min_value=0.0, step=1.0, value=st.session_state.nmiles)

if st.button("Predict Frontier Crew"):
    res = output(st.session_state.fare, st.session_state.nmiles)
    predicted_passengers = res[0]
    attendants = res[1]
        
    st.write(f"Predicted Number of Passengers: {predicted_passengers:.2f}")
    st.write(f"Predicted Number of Attendants: {attendants:.2f}")

## ------SPIRIT---------
# %%
plt.figure(figsize=(10, 6))
bins = [0, 500, 1000, 1500, 2000, 2500, 2724]
labels = ['0-500', '500-1000', '1000-1500', '1500-2000', '2000-2500', '2500-2724']

NK_df['nsmiles_binned'] = pd.cut(NK_df['nsmiles'], bins=bins, labels=labels, include_lowest=True)

print(NK_df[['nsmiles', 'nsmiles_binned']].head())

so.boxplot(x='nsmiles_binned', y='passengers', data = NK_df)
plt.title('Box Plot of Passengers vs Number of Miles')
plt.xlabel('Number of Miles')
plt.ylabel('Number of Passengers')

grouped = NK_df.groupby('nsmiles_binned')['passengers']
for i, group in enumerate(grouped):
    first = group[1].quantile(0.25) 
    median = group[1].quantile(0.5)
    third = group[1].quantile(0.75)  
    iqr = third - first  
    print(f"Bin {group[0]}: Q1 = {first}, Median = {median}, Q3 = {third}, IQR = {iqr}")
   
plt.show()



# %%
def output(input_fare, input_miles):
    predicted_passengers = predict_passengers(input_fare, input_miles)
    print(f"Predicted number of passengers: {predicted_passengers}")

    capacity_NK  = {
        500 : 509,
        1000 : 669,
        1500 : 800,
        2000 : 640,
        2500 : 560,
        2724 : 57
    }

    bucket = 0
    for key in capacity_NK:
        if nsmiles_input <= key:
            bucket = key
            break
    if predicted_passengers > capacity_NK.get(bucket):
        predicted_passengers = capacity_NK.get(bucket)
        
    print(f"Predicted number of passengers: {predicted_passengers}")

    crew  = {
        18 : 0,
        50 : 1,
        100 : 2
    }
    attendants = -1

    for key in crew:
        if predicted_passengers <= key:
            attendants = crew.get(key)
            break
        
    if attendants == -1:
        predicted_passengers -= 100
        attendants = predicted_passengers // 50 + 3
    print(f"Predicted number of attendants: {attendants}") 

    return [predicted_passengers, attendants]

# %%
st.subheader("Input Fare and Number of Miles for Crew Prediction - Spirit Airlines")

if 'fare' not in st.session_state:
    st.session_state.fare = 0.0
if 'nmiles' not in st.session_state:
    st.session_state.nmiles = 0.0

st.session_state.fare = st.number_input("Fare ($):", min_value=0.0, step=1.0, value=st.session_state.fare, key='fare_input')
st.session_state.nmiles = st.number_input("Number of Miles:", min_value=0.0, step=1.0, value=st.session_state.nmiles, key='miles_input')

if st.button("Predict Spirit Crew"):
    res = output(st.session_state.fare, st.session_state.nmiles)
    predicted_passengers = res[0]
    attendants = res[1]
        
    st.write(f"Predicted Number of Passengers: {predicted_passengers:.2f}")
    st.write(f"Predicted Number of Attendants: {attendants:.2f}")


# %%


