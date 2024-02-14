import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Define a function to read and preprocess the baseball data
def read_and_preprocess_data(directory):
    # List all files in the directory
    files = os.listdir(directory)

    # Create an empty DataFrame to store the data
    df_list = []

    # Iterate through each file in the directory
    for file in files:
        # Read the file into a DataFrame
        df = pd.read_excel(os.path.join(directory, file))

        # Drop columns without any observed values
        df = df.dropna(axis=1, how='all')

        # Convert all columns to numerical data type
        df = df.apply(pd.to_numeric, errors='coerce')

        # Drop columns ['VH', 'Team', 'Pitcher'] before imputation
        df = df.drop(columns=['VH', 'Team', 'Pitcher'])

        # Append the DataFrame to the list
        df_list.append(df)

    # Concatenate all DataFrames in the list
    combined_df = pd.concat(df_list, ignore_index=True)

    # Preprocess the data (e.g., handle missing values, feature engineering)
    # Replace "score_column" with the appropriate column name for the target variable
    combined_df['Score'] = combined_df['1st'] + combined_df['2nd'] + combined_df['3rd'] + combined_df['4th'] + combined_df['5th'] + combined_df['6th'] + combined_df['7th'] + combined_df['8th'] + combined_df['9th']
    combined_df = combined_df.drop(columns=['1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th'])

    # Impute missing values with the median of each column
    imputer = SimpleImputer(strategy='median')
    combined_df = pd.DataFrame(imputer.fit_transform(combined_df), columns=combined_df.columns)

    return combined_df

# Define the directory where the datasets are stored
directory = "./datasets/"

# Read and preprocess the datasets
baseball_data = read_and_preprocess_data(directory)

# Split data into training and testing sets
X = baseball_data.drop(columns=["Score"]) # Features
y = baseball_data["Score"] # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
