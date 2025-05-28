# APPLICATION OF MACHINE LEARNING IN PREDICTING TAIGA GOOSE MIGRATION
This project focuses on analyzing and predicting the migration patterns of the Taiga Bean Goose, using real-world GPS tracking data and machine learning. The main objective was to understand movement behavior and build a model to predict future locations of individual birds based on past movements.

The data was sourced from Movebank, containing GPS tracking records for 44 individual geese over the 2019–2020 migration period. Each record included latitude, longitude, timestamp, speed, heading, and altitude. Additional environmental variables like wind, temperature, and precipitation were integrated from the ERA5 climate dataset.

Project Workflow:
	1.	Data Cleaning & Preprocessing
	•	Removed missing or invalid coordinates
	•	Interpolated small temporal gaps
	•	Merged climate and tracking data
	2.	Exploratory Data Analysis (EDA)
	•	Visualized flight paths, altitude distributions, and speed trends
	•	Clustered birds by movement behavior
	•	Analyzed individual differences between birds
	3.	Predictive Modeling
	•	Used a Random Forest Regressor to predict future GPS coordinates
	•	Evaluated model accuracy with RMSE and MAE
	•	Applied SHAP to interpret feature importance
	4.	Interactive Dashboard (Locally Hosted)
	•	Built using Streamlit
	•	Features include:
	•	Animated and clustered migration maps
	•	Prediction vs actual visualization
	•	SHAP-based model interpretation
	•	Bird-specific filtering and insights

 Tools & Technologies:
	•	Python, Pandas, scikit-learn, Matplotlib, SHAP, Folium, Plotly
	•	Jupyter Notebook for experimentation
	•	Streamlit for dashboard development (hosted locally)
	•	GitHub for version control
	•	HTML5 UP for portfolio styling

Notes:
	•	The dashboard is locally hosted, meaning it runs on your machine via streamlit run app.py
	•	All visualizations, model files, and results are included in the GitHub repository
	•	A full report is available as a PDF explaining the methodology, results, and findings
