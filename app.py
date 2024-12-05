#Importing Libraries
from flask import Flask, render_template, request, Response
import plotly.express as px
import pandas as pd
import json
import plotly.graph_objects as go
from joblib import load
from sklearn.preprocessing import QuantileTransformer

#Initializing the Flask Application
app = Flask(__name__)

# File paths
geojson_path = 'data/map.geojson'
model_path = 'model.joblib'

# Load traffic data and preprocess time column
data_frame = pd.read_csv('data/Traffic.csv') 

# Load GeoJSON data
def load_geojson():
    with open(geojson_path) as f:
        return json.load(f)

# Load trained model
def load_model():
    return load(model_path)

scaler = QuantileTransformer(output_distribution='normal', n_quantiles=1)

# Data Preprocessing Function
def preprocess_data(data, hour, minute, day):
    filtered_data = data[(
            pd.to_datetime(data['Time'], format='%I:%M:%S %p').dt.hour == hour) & 
            (pd.to_datetime(data['Time'], format='%I:%M:%S %p').dt.minute == minute) & 
            (data['Date'] == day)
        ].copy()
    filtered_data['Weekend'] = filtered_data['Day of the week'].isin(['Saturday', 'Sunday'])
    filtered_data['Hour'] = pd.to_datetime(filtered_data['Time'], format='%I:%M:%S %p').dt.hour
    filtered_data['Traffic Situation'] = filtered_data['Traffic Situation'].astype('category').cat.codes
    filtered_data[['CarCount', 'BikeCount', 'BusCount', 'TruckCount']] = scaler.fit_transform(filtered_data[['CarCount', 'BikeCount', 'BusCount', 'TruckCount']])
    filtered_data = filtered_data.drop(columns=['Traffic Situation'])   

    return filtered_data

#Defining the Main Route
@app.route('/', methods=['GET', 'POST'])
def index():
    # Load geojson and model
    geojson_data = load_geojson()
    model = load_model()

    # Simulate choropleth data for the map
    data = {
        'name': ['Heavy', 'Low', 'Normal', 'High'],
        'value': [0, 1, 2, 3]
    }
    df = pd.DataFrame(data)

    # Plot the choropleth map
    fig = px.choropleth_mapbox(
        df,
        geojson=geojson_data,
        locations='name',
        color='name',
        mapbox_style='open-street-map',
        zoom=15,
        center={"lat": 13.284417, "lon": 74.748368},
        opacity=0.5,
        labels={'name': 'Traffic Situation'},
        width=800,
        height=500,
        color_discrete_map={
            "Heavy": "red",
            "Low": "green",
            "Normal": "blue",
            'High': 'yellow'
        },
    )

    # Add a line trace for Katpadi,Udupi
    line_data = pd.DataFrame({
        'lat': [13.287195, 13.280701],
        'lon': [74.747059, 74.749302],
        'name': ['Start', 'End']
    })
    fig.add_trace(go.Scattermapbox(
        lat=line_data['lat'],
        lon=line_data['lon'],
        mode='lines+markers',
        marker=go.scattermapbox.Marker(size=8, color='red'),
        line=dict(width=4, color='blue'),
        name='Katpadi'
    ))

    filtered_data_html = ""
    predicted_output = ""
    output = ''

    # Initialize video_index
    video_index = 0
    if request.args.get('video_index') is not None:
        video_index = int(request.args.get('video_index'))

    if request.method == 'POST':
        # Get user input from form
        day = int(request.form['day'])
        hour = int(request.form['hour'])
        am_pm = request.form['am_pm']
        minute = int(request.form['minute'])

        # Convert to 24-hour format
        if am_pm == 'PM' and hour < 12:
            hour += 12
        if am_pm == 'AM' and hour == 12:
            hour = 0

        input_features = preprocess_data(data_frame, hour=hour, minute=minute, day=day)

        # Check if data is available
        if not input_features.empty:
            input = input_features.iloc[0].to_dict()
            input_features_df = pd.DataFrame([input])

            predicted_output = model.predict(input_features_df)[0]
            inverse_transformed_data = scaler.inverse_transform(input_features[['CarCount', 'BikeCount', 'BusCount', 'TruckCount']])
            input_features[['CarCount', 'BikeCount', 'BusCount', 'TruckCount']] = inverse_transformed_data

            # Define color map for prediction output
            color_map = {'heavy': 'red', 'low': 'green', 'normal': 'blue', 'high': 'yellow'}
            output_map = {'heavy': 'Heavy', 'low': 'Low', 'normal': 'Normal', 'high': 'High'}
            line_color = color_map.get(predicted_output, 'gray')
            output = output_map.get(predicted_output, 'None')

            # Update line color in the map for prediction
            fig.data[-1].line.color = line_color

            # Convert filtered data to HTML table
            filtered_data_html = input_features.iloc[:, :-3].to_html(classes='table table-striped', index=False)
        else:
            predicted_output = "No data available for the selected time and date."
            line_color = 'gray'
            output = 'None'

    # Update the map with the new trace
    graph_html = fig.to_html(full_html=False)

    # Render the template with the prediction result and video index
    return render_template('index.html', graph_html=graph_html, filtered_data_html=filtered_data_html, output=output, video_index=video_index)

@app.route('/video_feed')
def video_feed():
    # This function should return a streaming response for your video feed.
    # For example purposes, let's just return a placeholder image.
    
    return Response("Video feed is not implemented yet.", mimetype='text/plain')

if __name__ == '__main__':
    app.run(debug=True)