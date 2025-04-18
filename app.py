import streamlit as st
import pandas as pd
import folium
from folium.plugins import HeatMap, MarkerCluster, TimestampedGeoJson, MiniMap, MousePosition
from streamlit_folium import st_folium
import joblib
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import pydeck as pdk
import gdown
import os

# Download files from Google Drive if not already present
files = {
    "nw_data.csv": "18Mau5WeoW2HeAmzIzxFbf_HeeNM0O3o5",
    "rf_features.pkl": "1TlBgB8ONKYJufOlqUm3s5bgfc0K0PlGL",
    "rf_model.pkl": "1MkyyBocQRtLrdg13p34bieEC3SDbYukW"
}

for filename, file_id in files.items():
    if not os.path.exists(filename):
        st.write(f"Downloading {filename} from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", filename, quiet=False)

# Load the data and models
df = pd.read_csv("nw_data.csv")
rf_features = joblib.load("rf_features.pkl")
rf_model = joblib.load("rf_model.pkl")

# --- Sidebar ---
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", [
    "EDA Snapshots",
    "Interactive Map",
    "Model Prediction",
    "Model Per Bird",
    "Compare Prediction vs Actual"
])
st.sidebar.markdown("---")
st.sidebar.title("Debug Info")
st.sidebar.write("Unique Location-Lat:", df['location-lat'].nunique())
st.sidebar.write("Unique Location-Long:", df['location-long'].nunique())

# --- Section 1: EDA Snapshots ---
if section == "EDA Snapshots":
    st.title("Exploratory Data Analysis Snapshots")
    
    st.image("images/eda_migration_paths.png", caption="Migration Paths")
    st.image("images/eda_migration_outliers.png", caption="Outliers Highlighted")
    st.image("images/eda_correlation_heatmap.png", caption="Correlation Heatmap")
    st.image("images/eda_altitude_dist.png", caption="Altitude Distribution")
    st.image("images/eda_heading_dist.png", caption="Heading Distribution")
    st.image("images/eda_temperature.png", caption="Temperature Over Time")
    st.image("images/eda_cluster_speed.png", caption="Speed per Cluster")
    st.image("images/eda_heatmap.png", caption="Migration Heatmap")
    st.image("images/eda_cluster_speed_over_time.png", caption="Speed Over Time")
    st.image("images/eda_cluster_direction_over_time.png", caption="Direction Over Time")
    st.image("images/eda_sensor_geolocation.png", caption="Sensor Movement")
    
    st.markdown("""
    ### Footnote:
    The above images represent various exploratory data analysis (EDA) aspects such as migration paths, 
    outliers, correlations, altitude distribution, temperature over time, and movement speed across clusters.
    These visualizations are crucial for understanding the bird migration patterns and the underlying dataset.
    """)
# --- Section 2: Interactive Map ---
elif section == "Interactive Map":
    st.title("🗺️ Interactive Migration Map")

    df_sample = df.sample(n=1000, random_state=42)

    if 'cluster' not in df_sample.columns:
        df_sample = df_sample.dropna(subset=['u10', 'v10'])
        kmeans = KMeans(n_clusters=3, random_state=42)
        df_sample['cluster'] = kmeans.fit_predict(df_sample[['u10', 'v10']])

    map_center = [df_sample['location-lat'].mean(), df_sample['location-long'].mean()]
    my_map = folium.Map(location=map_center, zoom_start=5, control_scale=True)
    marker_cluster = MarkerCluster(name="Migration Points").add_to(my_map)

    for _, row in df_sample.iterrows():
        color = ['blue', 'red', 'green'][int(row['cluster']) % 3]
        popup = f"""
        <b>Bird ID:</b> {row['individual-local-identifier']}<br>
        <b>Speed:</b> {row['u10']:.2f} m/s<br>
        <b>Direction:</b> {row['v10']:.2f}&deg;<br>
        <b>Timestamp:</b> {row['timestamp']}
        """
        folium.CircleMarker(
            location=[row['location-lat'], row['location-long']],
            radius=5,
            color=color,
            fill=True,
            fill_opacity=0.6,
            popup=popup
        ).add_to(marker_cluster)

    heat_data = [[row['location-lat'], row['location-long']] for _, row in df_sample.iterrows()]
    HeatMap(heat_data, radius=10, blur=15).add_to(my_map)

    features = [
        {
            'type': 'Feature',
            'geometry': {'type': 'Point', 'coordinates': [row['location-long'], row['location-lat']]},
            'properties': {
                'time': row['timestamp'].isoformat(),
                'popup': f"{row['individual-local-identifier']} - {row['u10']:.2f} m/s",
                'icon': 'circle',
                'iconstyle': {'fillColor': 'orange', 'radius': 4}
            }
        } for _, row in df_sample.iterrows()
    ]

    TimestampedGeoJson({"type": "FeatureCollection", "features": features},
                        period='PT1H', add_last_point=True, loop=False,
                        auto_play=False, time_slider_drag_update=True).add_to(my_map)

    MiniMap(toggle_display=True).add_to(my_map)
    MousePosition(position='bottomright').add_to(my_map)
    folium.LayerControl(collapsed=False).add_to(my_map)
    components.html(my_map._repr_html_(), height=800)
    st.markdown("""
    ### Footnote:
    This map visualizes the bird migration points, with color-coded markers indicating different clusters based 
    on movement characteristics like wind speed and direction. A heatmap overlays the migration path, and a 
    timestamped geojson shows the bird movement over time.
    """)

# --- Section 3: Model Prediction ---
elif section == "Model Prediction":
    st.title("📍 Predict Next Location")

    st.subheader("Input Features")
    input_data = {}

    for feature in feature_names:
        if feature in df.columns:
            default = float(df[feature].median())
            min_val = float(df[feature].min())
            max_val = float(df[feature].max())
        else:
            default = 0.0
            min_val = -100.0
            max_val = 100.0

        input_data[feature] = st.number_input(
            f"{feature}", value=default, min_value=min_val, max_value=max_val, key=f"input_{feature}"
        )

    input_df = pd.DataFrame([input_data])
    st.markdown("### 🔍 Model Input Preview")
    st.write(input_df)

    prediction = model.predict(input_df)
    st.markdown("### 📍 Predicted Next Location:")
    st.write(f"**Latitude:** {prediction[0][0]:.6f}")
    st.write(f"**Longitude:** {prediction[0][1]:.6f}")

    st.markdown("### 📈 Raw Prediction Output")
    st.write(prediction)

    # Display Map
    st.markdown("#### 🗺️ Bird's Predicted Location on Map")
    bird_map = folium.Map(location=[prediction[0][0], prediction[0][1]], zoom_start=12)
    folium.Marker(
        location=[prediction[0][0], prediction[0][1]],
        popup=f"Predicted Location\nLatitude: {prediction[0][0]:.6f}\nLongitude: {prediction[0][1]:.6f}",
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(bird_map)
    components.html(bird_map._repr_html_(), height=400)

    st.markdown("""
    ### Footnote:
    The model takes the input features such as wind components, speed, altitude, and temperature to predict 
    the bird's next location. The prediction is displayed on a map, which gives a spatial representation of 
    the predicted movement.
    """)


# --- Section 4: Model Per Bird ---

elif section == "Model Per Bird":
    st.title("Prediction per Bird")
    bird_ids = df['individual-local-identifier'].unique()
    selected_bird = st.selectbox("Select a Bird", bird_ids)
    bird_df = df[df['individual-local-identifier'] == selected_bird]

    st.write(f"Showing {len(bird_df)} records for bird: {selected_bird}")

    if st.button("Predict All for This Bird"):
        bird_input = bird_df[feature_names].dropna()
        bird_preds = model.predict(bird_input)

        bird_df_result = bird_df.loc[bird_input.index].copy()
        bird_df_result['pred_latitude'] = bird_preds[:, 0]
        bird_df_result['pred_longitude'] = bird_preds[:, 1]

        # Downsample for performance
        bird_df_result = bird_df_result.iloc[::5].copy()

        # Prepare actual and predicted data
        actual_df = bird_df_result[['timestamp', 'location-lat', 'location-long']].copy()
        actual_df.columns = ['timestamp', 'lat', 'lon']
        actual_df['type'] = 'Actual'

        predicted_df = bird_df_result[['timestamp', 'pred_latitude', 'pred_longitude']].copy()
        predicted_df.columns = ['timestamp', 'lat', 'lon']
        predicted_df['type'] = 'Predicted'

        combined_df = pd.concat([actual_df, predicted_df], ignore_index=True)

        # Add color and radius
        combined_df['color'] = combined_df['type'].map({'Actual': [0, 0, 255], 'Predicted': [255, 0, 0]})
        combined_df['radius'] = 1000  # meters

        # Create PathLayer for lines (optional)
        actual_path = actual_df[['lat', 'lon']].values.tolist()
        predicted_path = predicted_df[['lat', 'lon']].values.tolist()

        # Deck.gl map
        midpoint = (combined_df['lat'].mean(), combined_df['lon'].mean())
        view_state = pdk.ViewState(latitude=midpoint[0], longitude=midpoint[1], zoom=4, pitch=0)

        layers = [
            # Actual and Predicted Points
            pdk.Layer(
                "ScatterplotLayer",
                data=combined_df,
                get_position='[lon, lat]',
                get_color='color',
                get_radius='radius',
                pickable=True,
                tooltip=True,
            ),
            # Actual path
            pdk.Layer(
                "PathLayer",
                data=[{'path': actual_path}],
                get_path='path',
                get_color='[0, 0, 255]',
                width_scale=10,
                width_min_pixels=2,
            ),
            # Predicted path
            pdk.Layer(
                "PathLayer",
                data=[{'path': predicted_path}],
                get_path='path',
                get_color='[255, 0, 0]',
                width_scale=10,
                width_min_pixels=2,
            ),
        ]

        st.pydeck_chart(pdk.Deck(
            map_style='mapbox://styles/mapbox/light-v9',
            initial_view_state=view_state,
            layers=layers,
            tooltip={"text": "{type} Point\nLat: {lat}\nLon: {lon}\nTime: {timestamp}"}
        ))

        # Download option
        csv = bird_df_result.to_csv(index=False).encode()
        st.download_button("Download Bird Predictions", csv, "bird_predictions.csv", "text/csv")

        st.markdown("""
### Footnote:
This section uses an interactive Pydeck map to show actual (blue) and predicted (red) movement of the selected bird. 
Hover on points to see the timestamp and location. You can download the full prediction data as a CSV file.
""")

# --- Section 5: Comparing Prediction vs Actual ---
elif section == "Compare Prediction vs Actual":
    st.title("\U0001F4CA Compare Model Predictions vs Actual (with True Evaluation)")

    if 'location-lat' in df.columns and 'location-long' in df.columns:
        actual = df[['timestamp', 'location-lat', 'location-long']].copy()
        actual = actual.dropna()

        # Convert to date for stratified sampling
        actual['date'] = actual['timestamp'].dt.date

        # --- Date filtering UI ---
        st.subheader("Filter by Date Range")
        min_date = df['timestamp'].min().date()
        max_date = df['timestamp'].max().date()

        start_date, end_date = st.slider(
            "Select Date Range",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM-DD"
        )

        # --- Apply date filter ---
        filtered_actual = actual[
            (actual['timestamp'].dt.date >= start_date) &
            (actual['timestamp'].dt.date <= end_date)
        ]

        # --- Random sampling from filtered results ---
        sample_size = st.slider("Number of Samples to Compare", 10, min(100, len(filtered_actual)), 50)
        sampled_actual = filtered_actual.sample(n=sample_size, random_state=42)

        # Align with input features
        model_input = df[feature_names].loc[sampled_actual.index].dropna()
        sampled_actual = sampled_actual.loc[model_input.index]

        predictions = model.predict(model_input)

        sampled_actual['pred_latitude'] = predictions[:, 0]
        sampled_actual['pred_longitude'] = predictions[:, 1]
        sampled_actual['lat_error'] = sampled_actual['location-lat'] - sampled_actual['pred_latitude']
        sampled_actual['lon_error'] = sampled_actual['location-long'] - sampled_actual['pred_longitude']

        rmse_lat = np.sqrt(mean_squared_error(sampled_actual['location-lat'], sampled_actual['pred_latitude']))
        rmse_lon = np.sqrt(mean_squared_error(sampled_actual['location-long'], sampled_actual['pred_longitude']))
        mae_lat = mean_absolute_error(sampled_actual['location-lat'], sampled_actual['pred_latitude'])
        mae_lon = mean_absolute_error(sampled_actual['location-long'], sampled_actual['pred_longitude'])

        st.metric("Latitude RMSE", f"{rmse_lat:.4f}")
        st.metric("Longitude RMSE", f"{rmse_lon:.4f}")
        st.metric("Latitude MAE", f"{mae_lat:.4f}")
        st.metric("Longitude MAE", f"{mae_lon:.4f}")

        st.write("### Diverse Daily Sample - Prediction vs Actual Table")
        st.dataframe(sampled_actual[['timestamp', 'location-lat', 'pred_latitude', 'lat_error',
                                     'location-long', 'pred_longitude', 'lon_error']].head(20))

        st.write("### Latitude Error Distribution")
        plt.figure(figsize=(8, 4))
        plt.hist(sampled_actual['lat_error'], bins=30, color='skyblue')
        plt.xlabel("Latitude Error")
        plt.ylabel("Frequency")
        st.pyplot(plt.gcf())

        st.write("### Longitude Error Distribution")
        plt.figure(figsize=(8, 4))
        plt.hist(sampled_actual['lon_error'], bins=30, color='salmon')
        plt.xlabel("Longitude Error")
        plt.ylabel("Frequency")
        st.pyplot(plt.gcf())

        st.write("### Scatter Plot: Actual vs Predicted Latitude")
        plt.figure(figsize=(8, 6))
        plt.scatter(sampled_actual['location-lat'], sampled_actual['pred_latitude'], alpha=0.5, c='purple')
        plt.plot([sampled_actual['location-lat'].min(), sampled_actual['location-lat'].max()],
                 [sampled_actual['location-lat'].min(), sampled_actual['location-lat'].max()], 'k--', lw=2)
        plt.xlabel("Actual Latitude")
        plt.ylabel("Predicted Latitude")
        plt.title("Actual vs Predicted Latitude (Daily Sampled)")
        st.pyplot(plt.gcf())

        st.write("### Scatter Plot: Actual vs Predicted Longitude")
        plt.figure(figsize=(8, 6))
        plt.scatter(sampled_actual['location-long'], sampled_actual['pred_longitude'], alpha=0.5, c='green')
        plt.plot([sampled_actual['location-long'].min(), sampled_actual['location-long'].max()],
                 [sampled_actual['location-long'].min(), sampled_actual['location-long'].max()], 'k--', lw=2)
        plt.xlabel("Actual Longitude")
        plt.ylabel("Predicted Longitude")
        plt.title("Actual vs Predicted Longitude (Daily Sampled)")
        st.pyplot(plt.gcf())

        # Footnote
        st.markdown("""
        <hr>
        <sub>
        **Note**: The visualizations and error metrics are based on a randomly sampled subset of data from the selected date range.  
        RMSE (Root Mean Square Error) and MAE (Mean Absolute Error) are commonly used metrics to assess model accuracy.  
        The dashed diagonal lines in scatter plots represent the ideal 1:1 prediction line, helping visualize how close the predictions are to the actual values.
        </sub>
        """, unsafe_allow_html=True)

    else:
        st.warning("location-lat and location-long columns not found in dataset.")
        plt.plot([sampled_actual['location-long'].min(), sampled_actual['location-long'].max()],
                 [sampled_actual['location-long'].min(), sampled_actual['location-long'].max()],
                 color='gray', linestyle='--')
        st.pyplot(plt.gcf())
