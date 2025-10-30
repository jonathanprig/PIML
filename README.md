# PIML
This project develops a hybrid, interpretable framework that integrates physics-based resistance-capacitance (RC) models with machine learning regression algorithms to predict indoor thermal dynamics in buildings. The workflow demonstrates how domain knowledge can be embedded within data-driven models to enhance interpretability and accuracy


🔹 Key Features:

Implemented XGBoost and Neural Network (TensorFlow/Keras) regression models trained on time-series data (100k+ records) generated from validated ASHRAE 140 benchmark cases.

Conducted data preprocessing, scaling, and feature selection using scikit-learn, improving model generalisation and stability.

Achieved R² = 0.93 and MAE = 2.0 °C in high-mass cases, highlighting strong predictive performance.

Visualised model behaviour through residual analysis and feature importance plots, supporting explainable decision-making.

Includes modular code for RC network generation, data simulation, and ML pipeline automation.


🔹 Technical Stack:

Python, XGBoost, TensorFlow, Keras, scikit-learn, NumPy, Matplotlib, Pandas


🔹 Applications:

Energy performance forecasting • Model interpretability • Time-series regression • Data analytics • Physics-informed ML
