# -*- coding: utf-8 -*-
"""
Created on Wed Feb 30 16:36:10 2024

@author: Tolulope Israel

"""

import sklearn
#print(sklearn.__version__)


# Import required libraries
import streamlit as st
import numpy as np
import pickle

# Load the trained KMeans model from a file
kmeans_model = pickle.load(open("kmeans.pkl", 'rb'))


# Step 10: Predictive Function for New Customers
def clustering(age, avg_income, spending_score):
    kmeans_model = pickle.load(open("kmeans.pkl", 'rb'))
    new_customer = np.array([[avg_income, spending_score]])  # Updated to use income and spending score
    predicted_cluster = kmeans_model.predict(new_customer)
    cluster_mapping = {
        0: """Moderate Spenders (Age: Middle-aged, Income: Moderate, Spending: Average).

            This group is fairly balanced in terms of spending.
            They’re not extravagant but do make purchases regularly.
""",
        1: """High Spenders (Age: Young, Income: High, Spending: High). High-income but cautious spenders.
        
        They could be more inclined to save or only spend on items they perceive as valuable.
        """,
        2: """Cautious Savers (Age: Middle-aged, Income: High, Spending: Low). 
        
        This group exhibits a high income yet shows low spending behavior, indicating a preference for saving or selective purchasing.""",
        3: """Young Lifestyle Spenders (Age: Youngest, Income: Low, Spending: High). 
        
        Despite lower income, this group is willing to spend on experiences and products that resonate with them, especially lifestyle-related items.""",
        4: """Price-Sensitive (Age: Older, Income: Low, Spending: Low). 
        
        This price-sensitive and budget-conscious group is likely to be responsive to value deals and discounts."""
}

    return cluster_mapping.get(predicted_cluster[0], "Unknown Cluster")


# Set up the Streamlit app title
st.title("Shopmore Customer Clustering App")

# Add a subheader with instructions for user input
st.subheader("Enter the customer details:")

# Set up user input fields in a two-column layout for better readability

# First row with two columns for Age and Annual Income inputs
col1, col2 = st.columns(2)
with col1:
    # Input field for customer age, with min and max values and default value of 30
    st.subheader("Customer Age")
    age = st.number_input("Age", min_value=18, max_value=100, value=30)

with col2:
    # Input field for average income, in thousands of dollars, with a range and default value of 50
    st.subheader("Annual Income (k$)")
    avg_income = st.number_input("Annual Income (k$)", min_value=0, max_value=200, value=50)

# Second row with a single column for Spending Score input
col3, _ = st.columns([1, 1])  # Only use col3
with col3:
    # Input field for spending score, with a range of 1-100 and default value of 50
    st.subheader("Spending Score (1-100)")
    spending_score = st.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50)

# Add a button to trigger the clustering function when clicked
if st.button("Predict Cluster"):
    # Run the clustering function with user inputs and store the result
    predicted_cluster = clustering(age, avg_income, spending_score)
    
    # Display the result in a success message on the app
    st.success(f'Customer"s Category: {predicted_cluster}')
