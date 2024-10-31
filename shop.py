# -*- coding: utf-8 -*-
"""
Shopmore Customer Clustering Application

Created on Feb 30 16:36:10 2024

@author: Tolulope Israel
"""

# Import necessary libraries
import streamlit as st
import numpy as np
import pickle

# Load the trained KMeans model from a file
with open("kmeans.pkl", 'rb') as file:
    kmeans_model = pickle.load(file)

# Function to categorize a new customer based on clustering
def clustering(age, avg_income, spending_score):
    new_customer = np.array([[avg_income, spending_score]])  # Updated to use income and spending score
    predicted_cluster = kmeans_model.predict(new_customer)[0]
    
    # Mapping cluster labels to customer profiles
    cluster_mapping = {
        0: """Moderate Spenders (Age: Middle-aged, Income: Moderate, Spending: Average).

        This group is fairly balanced in terms of spending.
        They’re not extravagant but do make purchases regularly.
        """,
        1: """High Spenders (Age: Young, Income: High, Spending: High).

        High-income but cautious spenders. They could be more inclined 
        to save or only spend on items they perceive as valuable.
        """,
        2: """Cautious Savers (Age: Middle-aged, Income: High, Spending: Low).

        This group exhibits a high income yet shows low spending behavior, 
        indicating a preference for saving or selective purchasing.
        """,
        3: """Young Lifestyle Spenders (Age: Youngest, Income: Low, Spending: High).

        Despite lower income, this group is willing to spend on experiences 
        and products that resonate with them, especially lifestyle-related items.
        """,
        4: """Price-Sensitive (Age: Older, Income: Low, Spending: Low).

        This price-sensitive and budget-conscious group is likely to be responsive 
        to value deals and discounts.
        """
    }
    return cluster_mapping.get(predicted_cluster, "Unknown Category")

# Set up the Streamlit app interface
st.title("Shopmore Customer Clustering App")
st.subheader("Enter the customer details:")

# Input fields for Age, Annual Income, and Spending Score
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)

with col2:
    avg_income = st.number_input("Annual Income (k$)", min_value=0, max_value=200, value=50)

col3, _ = st.columns([1, 1])
with col3:
    spending_score = st.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50)

# Predict and display the customer category
if st.button("Predict Cluster"):
    predicted_category = clustering(age, avg_income, spending_score)
    st.success(f"Customer's Category: {predicted_category}")
