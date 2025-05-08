import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import folium
from folium.plugins import HeatMap

st.set_page_config(layout="wide")
st.title("Crime Incident Forecasting & Risk Dashboard")

# Download the csv from the source first.
df = pd.read_csv('https://www.dallasopendata.com/Public-Safety/Police-Incidents/qv6i-rri7/about_data')



# to be more efficient, we copy the dataset to be used in memory
data = df.copy()

# Extract Latitude and Longitude
def extract_lat_lon(location_str):
    match = re.search(r'\(([^,]+),\s*([^)]+)\)', str(location_str))
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None

data[['Latitude', 'Longitude']] = data['Location1'].apply(lambda x: pd.Series(extract_lat_lon(x)))

# Parse date and time
data['Date1 of Occurrence'] = pd.to_datetime(data['Date1 of Occurrence'], errors='coerce')
data['Month'] = data['Date1 of Occurrence'].dt.month
data['DayOfWeek'] = data['Date1 of Occurrence'].dt.dayofweek
data['Hour'] = pd.to_datetime(data['Time1 of Occurrence'], errors='coerce').dt.hour

# Encode incident type
le_type = LabelEncoder()
data['IncidentCategory'] = le_type.fit_transform(data['Type of Incident'].astype(str))
model_data = data[['Latitude', 'Longitude', 'Month', 'DayOfWeek', 'Hour', 'IncidentCategory']].dropna()

# KDE Heatmap
st.subheader("Crime Risk Heatmap (KDE)")
fig, ax = plt.subplots(figsize=(8, 6))
sns.kdeplot(data=model_data, x="Longitude", y="Latitude", fill=True, cmap="Reds", ax=ax, thresh=0.05, levels=100)
st.pyplot(fig)

# Classification Model
st.subheader("Crime Type Prediction Report")
X = model_data[['Latitude', 'Longitude', 'Month', 'DayOfWeek', 'Hour']]
y = model_data['IncidentCategory']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=False, zero_division=0)
st.text(report)

# Forecasting
st.subheader("Monthly Incident Forecast For The Next 6 Months")
data['IncidentMonth'] = data['Date1 of Occurrence'].dt.to_period('M')
monthly_counts = data.groupby('IncidentMonth').size().rename('IncidentCount').to_timestamp()
monthly_counts = monthly_counts.dropna()
if len(monthly_counts) > 1:
    model = ExponentialSmoothing(monthly_counts, trend='additive', seasonal=None, initialization_method="estimated")
    fit_model = model.fit()
    forecast = fit_model.forecast(6)
    st.line_chart(pd.concat([monthly_counts, forecast]))
else:
    st.warning("Not enough data for time-based forecasting.")


# Now, we are creating a folium map centered around the average location
center_lat = model_data['Latitude'].mean()
center_lon = model_data['Longitude'].mean()
map_heatmap = folium.Map(location=[center_lat, center_lon], zoom_start=12)

# Here, we are preparing the data for heatmap
heat_data = model_data[['Latitude', 'Longitude']].values.tolist()

# We are adding a heatmap layer
HeatMap(heat_data).add_to(map_heatmap)

# We are saving our result heatmap to HTML
output_path = "Police_Incidents_heatmap.html"
map_heatmap.save(output_path)

output_path

## Please check the current directory for the .html file to view the interactive results.

st.title("The Iteractive HeatMap")
map_heatmap.show_in_browser()


