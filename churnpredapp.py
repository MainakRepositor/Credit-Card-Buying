import numpy as np
import pandas as pd
import streamlit as st
import pickle
import sys

with st.sidebar:
   st.title("Credit Card Churn Prediction")

   st.info("Prediction Estimates:")

dependentvarcat_dict = pickle.load(open("dependentvariable_catcodes.pkl", "rb"))
final_collst = pickle.load(open("final_collst.pkl", "rb") )
mlmodel = pickle.load(open("xgboostmodel.pkl", "rb"))
card_category_ohe = pickle.load(open("card_category.pkl","rb"))
education_level_ohe = pickle.load(open("education_level.pkl","rb"))
gender_ohe = pickle.load(open("gender.pkl","rb"))
income_category_ohe = pickle.load(open("income_category.pkl","rb"))
marital_status_ohe = pickle.load(open("marital_status.pkl","rb"))
minmax_scaler = pickle.load(open("minmax_scaler.pkl","rb"))


st.header("Credit Card Churn Prediction")
st.markdown("Please fill all the below information")

with st.expander("Personal Details"):
   customer_age = st.number_input("Age: ", min_value=18,step=1)
   gender = st.radio(
      "Gender: ",
      ["F", "M"],
   )
   dependent_count = st.number_input("Number of Dependent: ", step=1)
   education_level = st.radio(
      "Education: ",
      ["Doctorate", "Post-Graduate", "Graduate", "College", "High School", "Uneducated", "Unknown"]
   )
   marital_status = st.radio(
      "Marital Status: ",
      ["Married", "Single", "Divorced", "Unknown"]
   )
   income_category = st.radio(
      "Annual Income Category: ",
      ["Less than $40K", "$40K - $60K", "$60K - $80K", "$80K - $120K", "$120K +", "Unknown"]
   )
   card_category = st.radio(
      "Type of Card: ",
      ["Blue", "Silver", "Gold", "Platinum"]
   )

with st.expander("Activity Details"):
   months_on_book = st.number_input("Period of Relationship: ", step=1)
   total_relationship_count = st.number_input("Total Number of Products: ", min_value=1, step=1)
   months_inactive_12_mon = st.number_input("Number of Months Inactive in the Last 12 Months: ", step=1)
   contacts_count_12_mon = st.number_input("Number of Contacts between the Customer and Bank in the Last 12 Months: ",
                                           step=1)

   credit_limit = st.number_input("Credit Limit: ")
   total_revolving_bal = st.number_input("Total Revolving balance: ", step=1)
   avg_open_to_buy = st.number_input("Average of Last 12 Months Amount Left on the Credit Card to Use: ")
   total_amt_chng_q4_q1 = st.number_input(
      "Ratio of the Total Transaction Amount in Fourth Quarter and the Total Transaction Amount in First Quarter: ")
   total_trans_amt = st.number_input("Total Transaction Amount in Last 12 Months: ", step=1)

   total_trans_ct = st.number_input("Total Transaction Count in Last 12 Months: ", step=1)
   total_ct_chng_q4_q1 = st.number_input(
      "Ratio of the Total Transaction Count in Fourth Quarter and the Total Transaction Count in First Quarter: ")
   avg_utilization_ratio = st.number_input("Available Credit Spent by the Customer: ")
   # st.button("Customer Churn Prediction", use_container_width=True)
   # st.divider()

if st.button("Customer Churn Prediction", use_container_width=True):
   independentvar = np.expand_dims(
      [total_trans_ct, total_revolving_bal, total_relationship_count, total_trans_amt, total_ct_chng_q4_q1, months_inactive_12_mon, total_amt_chng_q4_q1, contacts_count_12_mon, customer_age, avg_open_to_buy], 0)
   independentvar = pd.DataFrame(independentvar, columns = final_collst)
   independentvar = minmax_scaler.transform(independentvar)
   independentvar = pd.DataFrame(independentvar, columns=final_collst)
   st.divider()


   y_pred = mlmodel.predict(independentvar)

   if list(dependentvarcat_dict.keys())[list(dependentvarcat_dict.values()).index(y_pred)] == 'Attrited Customer':
      st.sidebar.success("The customer is likely to buy Credit Card")
   else:
      st.sidebar.error("The customer is least likely to buy Credit Card")


   
   

