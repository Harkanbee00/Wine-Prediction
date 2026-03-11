import joblib
import streamlit as st
import numpy as np
import pandas as pd




model = joblib.load("rrforestwine.pkl")



st.title("Wine Classification Application")


st.write("Predict Wine Quality Score Using a Random Forest Model")


form = st.form("wine Quality form")

form.subheader("Enter Score")

fixed_acidity = form.slider(

		"Fixed_acidity ",
		min_value= 4.0,
		max_value= 12.0,
		value = 7.1	

	)

volatile_acidity = form.slider(

		"volatile acidity",
		min_value= 0.0,
		max_value= 1.0,
		value = 0.2

	)

citric_acid = form.slider(

		"citric acid",
		min_value= 0.0,
		max_value= 1.5,
		value = 0.2

	)

residual_sugar = form.slider(

		"residual sugar",
		min_value= 1.0,
		max_value= 5.0,
		value = 1.4

	)
chlorides = form.slider(

		"chlorides ",
		min_value= 0.005,
		max_value= 0.10,
		value = 0.073	

	)

free_sulfur_dioxide = form.slider(

		"free sulfur dioxide",
		min_value= 5.0,
		max_value= 30.0,
		value = 13.2

	)

total_sulfur_dioxide = form.slider(

		"total sulfur dioxide",
		min_value= 20.0,
		max_value= 100.0,
		value = 30.0
	)
density = form.slider(

		"density ",
		min_value= 0.0001,
		max_value= 1.0000,
		value = 0.9950	

	)

pH = form.slider(

		"pH acidity",
		min_value= 2.0,
		max_value= 5.0,
		value = 3.0

	)

sulphates = form.slider(

		"sulphates",
		min_value= 0.1,
		max_value= 1.,
		value = 0.55

	)

alcohol = form.slider(

		"alcohol",
		min_value = 5.0,
		max_value = 10.0,
		value= 9.0

	)


submit_button = form.form_submit_button("Predict")


iif submit_button:
    input_data = [[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                   chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                   density, pH, sulphates, alcohol]]
    prediction = model.predict(input_data)
    
    st.subheader("Prediction Result")
    
    if prediction[0] == 1:
        st.success("Predicted Quality: Good Quality")   # green
    else:
        st.error("Predicted Quality: Bad Quality")      # red











