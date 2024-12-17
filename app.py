import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load the model and transformer
with open('final_model_rf.pkl', 'rb') as file:
    model = pickle.load(file)

with open('transformer.pkl', 'rb') as file:
    pt = pickle.load(file)

# Define the prediction function
def prediction(input_list):
    input_list = np.array(input_list,dtype='object')
    tran_data = pt.transform([[input[0], input[3]])
    input[0] = trans_data[0][0]
    input[3] = trans_data[0][1]

    pred = model.predict_proba([input_list])[:,1][0]
    if pred > 0.5:
        return f'This booking is more likely to get cancelled: chances {round(pred, 2)}'
    else:
        return f'This booking is less likely to get cancelled: chances {round(pred, 2)}'

# Define the main function
def main():
    st.title('INN HOTEL GROUPS')

    lt_input = st.text_input('Enter the lead time.')

    mst = (lambda x: 1 if x == 'Online' else 0)(st.selectbox('Choose the type of booking', ['Online', 'Offline']))
    spcl = st.selectbox('Select the no. of special requests made', [0, 1, 2, 3, 4, 5])
    
    price_input = st.text_input('Enter the price offered for the room.')
    
    adult = st.radio('Select the number of adults in booking', [0, 1, 2, 3, 4])

    wkd_input = st.text_input('Enter the number of weekend nights in the booking')

    wk_input = st.text_input('Enter the number of week nights in the booking')

    park = (lambda x: 1 if x == 'Yes' else 0)(st.selectbox('Is parking included in the booking', ['Yes', 'No']))
    month = st.slider('What will be the month of arrival', min_value=1, max_value=12, step=1)
    day = st.slider('What will be the day of arrival', min_value=1, max_value=31, step=1)

    wkday_lambda = (lambda x: 0 if x == 'Mon' else 1 if x == 'Tue' else 2 if x == 'Wed' else 3 if x == 'Thu' else 4 if x == 'Fri' else 5 if x == 'Sat' else 6)
    wkday = wkday_lambda(st.selectbox('What is the weekday of arrival', ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']))

    trans_data = pt.transform([[lt_input, price_input]])
    lt_t = trans_data[0][0]
    price_t = trans_data[0][1]

    inp_list = [lt_t, mst, spcl, price_t, adult, wkd_input, wk_input, park, month, day, wkday]

    if st.button('Predict'):
        response = prediction(inp_list)
        st.success(response)

if __name__ == '__main__':
    main()
