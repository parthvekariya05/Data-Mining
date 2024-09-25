import pandas as pd
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import os
# Function to preprocess the data
def preprocess_data(data):
    # Encode categorical columns (non-numeric)
    label_encoders = {}
    
    for column in data.columns:
        if data[column].dtype == 'object':
            le = LabelEncoder()
            data[column] = le.fit_transform(data[column].astype(str))
            label_encoders[column] = le
    return data, label_encoders

# Function to load and process the CSV or Excel file
def load_file():
    file_path = filedialog.askopenfilename(
        filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")])
    if file_path:
        try:
            if file_path.endswith('.csv'):
                data = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                data = pd.read_excel(file_path)
            else:
                messagebox.showerror("File Error", "Unsupported file format!")
                return
            
            # Drop rows where the target column (last column) contains NaN
            data = data.dropna(subset=[data.columns[-1]])

            # If the dataset is empty after dropping NaN rows, show an error
            if data.empty:
                messagebox.showerror("Error", "No data available after removing rows with missing target values.")
                return

            # Preprocess the data
            data, label_encoders = preprocess_data(data)
            
            # Extract target variable and features (assuming the last column is the target)
            X = data.iloc[:, :-1]  # Features
            y = data.iloc[:, -1]   # Target
            
            # Check if there's enough data to split
            if len(data) < 2:
                messagebox.showerror("Error", "Not enough data to train the model.")
                return
            
            # Split the dataset, adjusting test_size if necessary
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train the decision tree
            clf = DecisionTreeClassifier()
            clf.fit(X_train, y_train)

            # Convert class names to a list of strings
            class_names = [str(cls) for cls in set(y)]

            # Display the decision tree
            plt.figure(figsize=(12,8))
            plot_tree(clf, feature_names=X.columns, class_names=class_names, filled=True)
            plt.show()

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

# Set up the tkinter UI
root = tk.Tk()
root.title("Decision Tree Generator")

# Add a button to upload file
upload_button = tk.Button(root, text="Upload CSV/Excel", command=load_file)
upload_button.pack(pady=20)

# Start the tkinter loop
root.geometry("300x200")
root.mainloop()