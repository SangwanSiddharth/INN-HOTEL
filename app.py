import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load the pre-trained model and transformer
with open('final_model_rf.pkl', 'rb') as file:
    model = pickle.load(file)

with open('transformer.pkl', 'rb') as file:
    pt = pickle.load(file)

def prediction(input_list):
    tran_data = pt.transform([[input_list[0], input_list[3]]])
    input_list[0] = tran_data[0][0]
    input_list[3] = tran_data[0][1]

    input_list = np.array(input_list, dtype=object)
    
    # Make prediction
    pred = model.predict_proba([input_list])[:, 1][0]
    
    # Return prediction result
    if pred > 0.5:
        return f'This booking is more likely to get canceled: Chances {round(pred, 2)}'
    else:
        return f'This booking is less likely to get canceled: Chances {round(pred, 2)}'
    
def main():
    st.title('INN HOTEL GROUP')
    
    # Use number input for numeric fields like lead time and price
    lt = st.number_input('Enter the lead time.', min_value=0.0)
    mst = (lambda x: 1 if x == 'Online' else 0)(st.selectbox('Enter the type of booking', ['Online', 'Offline']))
    spcl = st.selectbox('Select the no of special requests made', [0, 1, 2, 3, 4, 5])
    price = st.number_input('Enter the price offered for the room', min_value=0.0)
    adults = st.selectbox('Select the no adults in booking', [0, 1, 2, 3, 4])
    weekend = st.number_input('Enter the weekend nights in the booking', min_value=0)
    weekday = st.number_input('Enter the week nights in booking', min_value=0)
    parking = (lambda x: 1 if x == 'Yes' else 0)(st.selectbox('Is parking included in the booking', ['Yes', 'No']))
    month = st.slider('What will be month of arrival', min_value=1, max_value=12, step=1)
    day = st.slider('What will be day of arrival', min_value=1, max_value=31, step=1)
    wkday_lambda = (lambda x: 0 if x == 'Mon' else 1 if x == 'Tue' else 2 if x == 'Wed' else 3 if x == 'Thus' 
                    else 4 if x == 'Fri' else 5 if x == 'Sat' else 6)
    weekd = wkday_lambda(st.selectbox('What is the weekday of arrival', ['Mon', 'Tue', 'Wed', 'Thus', 'Fri', 'Sat', 'Sun']))

    inp_list = [lt, mst, spcl, price, adults, weekend, parking, weekday, month, day, weekd]
    
    if st.button('Predict'):
        if lt == 0.0 or price == 0.0:  # Check if lead time and price are provided
            st.error("Lead time and price cannot be empty or zero.")
        else:
            try:
                # Make the prediction
                response = prediction(inp_list)
                st.success(response)
            except ValueError:
                st.error("An error occurred during the prediction. Please check your inputs.")
        
if __name__ == '__main__':
    main()
