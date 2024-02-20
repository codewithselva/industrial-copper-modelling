import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

# Read the CSV file and load it into a Pandas DataFrame
excel_file_path = 'copper_data_set.csv'

df = pd.read_csv(excel_file_path)


df.head()

# Create a copy to avoid modifying the original DataFrame
cleaned_df = df.copy()

cleaned_df = cleaned_df[(cleaned_df['status'] == 'Won') | (cleaned_df['status'] == 'Lost')]


# Assuming 'your_column' is the column you're working with
cleaned_df['quantity tons'] = pd.to_numeric(cleaned_df['quantity tons'], errors='coerce')

cleaned_df['item_date'] = pd.to_datetime(cleaned_df['item_date'], format='%Y%m%d', errors='coerce')
cleaned_df['delivery date'] = pd.to_datetime(cleaned_df['delivery date'], format='%Y%m%d', errors='coerce')


# Some rubbish values are present in ‘Material_Reference’ which starts with ‘00000’ value which should be converted into null
cleaned_df['material_ref'] = cleaned_df['material_ref'].apply(lambda x: None if str(x).startswith('00000') else x)

cleaned_df['customer'] = cleaned_df['customer'].astype(pd.Int64Dtype())


# Checking for consistency in categorization
cleaned_df['country'] = cleaned_df['country'].astype('category')
cleaned_df['status'] = cleaned_df['status'].astype('category')
cleaned_df['item type'] = cleaned_df['item type'].astype('category')
cleaned_df['application'] = cleaned_df['application'].astype('category')
cleaned_df['product_ref'] = cleaned_df['product_ref'].astype('category')
cleaned_df['material_ref'] = cleaned_df['material_ref'].astype('category')
cleaned_df['customer'] = cleaned_df['customer'].astype('category')


# Handle missing values (replace with a default value or fill using a specific strategy)
cleaned_df['quantity tons'].fillna(0, inplace=True) 
cleaned_df['thickness'].fillna(0, inplace=True) 
cleaned_df['width'].fillna(0, inplace=True) 
cleaned_df['selling_price'].fillna(0, inplace=True) 


skewness = cleaned_df[cleaned_df.select_dtypes(include=['number']).columns].skew()

# Display skewness for each numerical column
print("Skewness for each numerical column:")
print(skewness)


outliers = (cleaned_df[cleaned_df.select_dtypes(include=['number']).columns] - cleaned_df[cleaned_df.select_dtypes(include=['number']).columns].mean()).abs() > 3 * cleaned_df[cleaned_df.select_dtypes(include=['number']).columns].std()  # Define your outlier detection method


# Data preprocessing
# Assume 'Selling_Price' is the target variable for regression
# Assume 'STATUS' is the target variable for classification

# Regression
regression_features = cleaned_df.drop(['selling_price', 'status','id','item_date','delivery date','material_ref', 'customer','item type','product_ref'], axis=1)
regression_target = cleaned_df['selling_price']


# Classification
classification_features = cleaned_df.drop(['selling_price','id','item_date','delivery date','material_ref', 'customer','item type','product_ref'], axis=1)
classification_target = cleaned_df['status']


from sklearn.model_selection import train_test_split

# Train-test split
regression_X_train, regression_X_test, regression_y_train, regression_y_test = train_test_split(
    regression_features, regression_target, test_size=0.2, random_state=42
)

classification_X_train, classification_X_test, classification_y_train, classification_y_test = train_test_split(
    classification_features, classification_target, test_size=0.2, random_state=42
)


from sklearn.preprocessing import StandardScaler

# Data normalization and feature scaling
scaler = StandardScaler()
regression_X_train_scaled = scaler.fit_transform(regression_X_train)
regression_X_test_scaled = scaler.transform(regression_X_test)

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
import streamlit as st
# Regression model
regression_model = RandomForestRegressor()
regression_model.fit(regression_X_train_scaled, regression_y_train)
regression_predictions = regression_model.predict(regression_X_test_scaled)
regression_rmse = np.sqrt(mean_squared_error(regression_y_test, regression_predictions))

# Classification model
classification_model = RandomForestClassifier()
classification_model.fit(classification_X_train, classification_y_train)
classification_predictions = classification_model.predict(classification_X_test)
classification_accuracy = accuracy_score(classification_y_test, classification_predictions)

# Streamlit App
st.title("Copper Industry ML Application")

# Sidebar for user input
st.sidebar.title("Insert Column Values")
user_input = {}
for column in df.columns:
    user_input[column] = st.sidebar.text_input(f"Enter {column}", df[column].iloc[0])

# Predictions
regression_input = pd.DataFrame([user_input])
classification_input = pd.DataFrame([user_input.drop(['Selling_Price'])])

# Scaling and prediction for regression
regression_input_scaled = scaler.transform(regression_input)
predicted_selling_price = regression_model.predict(regression_input_scaled)

# Prediction for classification
predicted_status = classification_model.predict(classification_input)[0]

# Display predictions
st.header("Regression Prediction (Selling Price)")
st.write(f"The predicted Selling Price is: {predicted_selling_price[0]}")

st.header("Classification Prediction (Status)")
st.write(f"The predicted Status is: {predicted_status}")

# Display model evaluation metrics
st.header("Model Evaluation Metrics")
st.write(f"Regression RMSE: {regression_rmse}")
st.write(f"Classification Accuracy: {classification_accuracy}")