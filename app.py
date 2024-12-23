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
    # Convert lead time and price to float
    lead_time = float(input_list[0])
    price = float(input_list[3])

    # Transform the lead time and price
    tran_data = pt.transform([[lead_time, price]])
    input_list[0] = tran_data[0][0]
    input_list[3] = tran_data[0][1]

    # Convert the input list to a numpy array
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
    lt = st.text_input('Enter the lead time.')
    mst = (lambda x: 1 if x == 'Online' else 0)(st.selectbox('Enter the type of booking', ['Online', 'Offline']))
    spcl = st.selectbox('Select the no of special requests made', [0, 1, 2, 3, 4, 5])
    price = st.text_input('Enter the price offered for the room')
    adults = st.selectbox('Select the no adults in booking', [0, 1, 2, 3, 4])
    weekend = st.text_input('Enter the weekend nights in the booking')
    weekday = st.text_input('Enter the week nights in booking')
    parking = (lambda x: 1 if x == 'Yes' else 0)(st.selectbox('Is parking included in the booking', ['Yes', 'No']))
    month = st.slider('What will be month of arrival', min_value=1, max_value=12, step=1)
    day = st.slider('What will be day of arrival', min_value=1, max_value=31, step=1)
    wkday_lambda = (lambda x: 0 if x == 'Mon' else 1 if x == 'Tue' else 2 if x == 'Wed' else 3 if x == 'Thus' 
                    else 4 if x == 'Fri' else 5 if x == 'Sat' else 6)
    weekd = wkday_lambda(st.selectbox('What is the weekday of arrival', ['Mon', 'Tue', 'Wed', 'Thus', 'Fri', 'Sat', 'Sun']))

    inp_list = [lt, mst, spcl, price, adults, weekend, parking, weekday, month, day, weekd]
    
    if st.button('Predict'):
        if lt == '' or price == '':
            st.error("Lead time and price cannot be empty.")
        else:
            try:
                # Convert inputs to appropriate types
                inp_list[0] = float(lt)
                inp_list[3] = float(price)
                response = prediction(inp_list)
                st.success(response)
            except ValueError:
                st.error("Please enter valid numeric values for lead time and price.")
        
if __name__ == '__main__':
    main()
