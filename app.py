import streamlit as st 
import pandas as pd
import joblib



model = joblib.load("rrforestwine.pkl")



st.title("Wine Classification Application")


st.write("Predict Wine Quality Score Using a Random Forest Model")


form = st.form("wine Quality form")

form.subheader("Enter Score")

Fixed acidity = form.number_input(

		"Fixed_acidity ",
		min_value= 4.0,
		max_value= 12.0,
		value = 7.1	

	)

volatile acidity = form.number_input(

		"volatile acidity",
		min_value= 0.0,
		max_value= 1.0,
		value = 0.2

	)

citric acid = form.number_input(

		"citric acid",
		min_value= 0.0,
		max_value= 1.5,
		value = 0.2

	)

residual sugar = form.number_input(

		"residual sugar",
		min_value= 1.0,
		max_value= 5.0,
		value = 1.4

	)
chlorides = form.number_input(

		"chlorides ",
		min_value= 0.005,
		max_value= 0.10,
		value = 0.073	

	)

free sulfur dioxide = form.number_input(

		"free sulfur dioxide",
		min_value= 5.0,
		max_value= 30.0,
		value = 13.2

	)

total sulfur dioxide = form.number_input(

		"total sulfur dioxide",
		min_value= 20.0,
		max_value= 100.0,
		value = 30.0
	)
density = form.number_input(

		"density ",
		min_value= 0.0001,
		max_value= 1.0000,
		value = 0.9950	

	)

pH = form.number_input(

		"pH acidity",
		min_value= 2.0,
		max_value= 5.0,
		value = 3.0

	)

sulphates = form.number_input(

		"sulphates",
		min_value= 0.1,
		max_value= 1.,
		value = 0.55

	)

alcohol = form.number_input(

		"alcohol",
		min_value = 5.0,
		max_value = 10.0,
		value= 9.0

	)


submit_button = form.form_submit_button("Predict")


if submit_button:
	input_data = [["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]]

	prediction = model.predict(input_data)


	st.subheader("Prediction Result")
	st.success(f" Predicted species: {prediction[0]}")