# COMP-4447-W25-MET_DATA_ANALYSIS
The Met Data Analysis Reid Holben &amp; Gabe Barela – Tools 1 (Winter 2025)

# MET Data Analysis

## Authors
Reid Holben and Gabriel Barela

## Description
This project involves loading and analyzing a dataset from the Metropolitan Museum of Art (MET) using Python libraries such as pandas, seaborn, and matplotlib. The dataset is hosted on GitHub and contains metadata about various objects in the MET collection.

## Installation
To install the necessary dependencies, run the following command:
```bash
! pip install -r requirements.txt
```
1. **Import Libraries**: Essential for data manipulation and analysis.
  ```python
  import pandas as pd
  import seaborn as sns
  import matplotlib.pyplot as plt
  import re
  ```
2. **Load the Dataset:**: Define the URL and load the dataset into a pandas DataFrame.
  ```python
  url = 'https://github.com/metmuseum/openaccess/raw/refs/heads/master/MetObjects.csv'
  data = pd.read_csv(url)
  ```
3. **Data Analysis:**: Perform descriptive statistics and other analyses.
  ```python
  print(data.describe())
  print(data.shape)
  ```
4. **Analyze Missing Data:**: Analyze missing data and combine it with the data types of each feature.
  ```python
  nullData = []
  nullCounts = data.isnull().sum()
  totalInstances = len(data)
  nullPercentages = (nullCounts / totalInstances) * 100
  
  for feature, count, percentage in zip(nullCounts.index, nullCounts, nullPercentages):
      nullData.append((feature, count, round(percentage, 2)))
  
  nullDataFrame = pd.DataFrame(nullData, columns=['Feature', 'Count', 'Percentage'])
  
  # Sort the DataFrame by 'Percentage' in descending order
  nullDataFrame = nullDataFrame.sort_values(by='Percentage', ascending=False)
  
  # Get the data types of each feature in the DataFrame
  data_types = data.dtypes
  
  # Convert the data types to a DataFrame for better readability
  data_types_df = data_types.reset_index()
  data_types_df.columns = ['Feature', 'Data Type']
  
  # Merge the null data and data types into a single DataFrame
  combined_df = pd.merge(nullDataFrame, data_types_df, on='Feature')
  
  combined_df
  ```
5. **Data Cleaning:** Clean the dataset by removing rows with NaN values in the 'AccessionYear' column and filtering out empty strings and zeros.
  ```python
  # Remove rows with NaN values in 'AccessionYear'
  data = data.dropna(subset=['AccessionYear'])
  data = data[(data['AccessionYear'] != '') & (data['AccessionYear'] != 0)]
  data.shape
  
  # List of features to keep for analysis of accession and object dates
  features_to_keep = ['AccessionYear', 'Object Begin Date', 'Object End Date', 'Department', 'Artist Begin Date', 'Artist End Date', 'Dimensions', 'Object Date']
  
  # Drop irrelevant features
  cleaned_data = data[features_to_keep]
  
  # Convert 'AccessionYear' to numeric, coercing errors to NaN
  cleaned_data.loc[:, 'AccessionYear'] = pd.to_numeric(cleaned_data['AccessionYear'], errors='coerce')
  
  # Drop rows where 'AccessionYear' is NaN after the conversion
  cleaned_data = cleaned_data.dropna(subset=['AccessionYear'])
  
  cleaned_data.head()
  cleaned_data.shape
  ```
6. **Plotting Distributions:** Visualize the distribution of dates by department.
  ```python
  # Plot the distribution of accession dates by department
  plt.figure(figsize=(12, 6))
  cleaned_data.groupby('Department')['AccessionYear'].plot(kind='hist', alpha=0.5, legend=True)
  plt.title('Distribution of Accession Dates by Department')
  plt.xlabel('Accession Year')
  plt.ylabel('Frequency')
  plt.legend(title='Department', bbox_to_anchor=(1.05, 1), loc='upper left')
  plt.show()
  
  # Plot the distribution of object begin dates by department
  plt.figure(figsize=(12, 6))
  cleaned_data.groupby('Department')['Object Begin Date'].plot(kind='hist', alpha=0.5, legend=True)
  plt.title('Distribution of Object Begin Dates by Department')
  plt.xlabel('Object Begin Date')
  plt.ylabel('Frequency')
  plt.legend(title='Department', bbox_to_anchor=(1.05, 1), loc='upper left')
  plt.show()
  
  # Plot the distribution of object end dates by department
  plt.figure(figsize=(12, 6))
  cleaned_data.groupby('Department')['Object End Date'].plot(kind='hist', alpha=0.5, legend=True)
  plt.title('Distribution of Object End Dates by Department')
  plt.xlabel('Object End Date')
  plt.ylabel('Frequency')
  plt.legend(title='Department', bbox_to_anchor=(1.05, 1), loc='upper left')
  plt.show()
  ```
7. **Anomaly Detection:** Detect anomalies in the dataset using the Isolation Forest algorithm.
  ```python
  import pandas as pd
  from sklearn.ensemble import IsolationForest
  import re
  
  # Convert categorical data to numerical data using one-hot encoding
  cleaned_data = pd.get_dummies(cleaned_data, columns=['Department'])
  
  # Ensure all columns are numeric
  for column in cleaned_data.columns:
      cleaned_data[column] = pd.to_numeric(cleaned_data[column], errors='coerce')
  
  # Save version pre one hot encoding
  cleaned_data_no_one = cleaned_data.copy()
  
  # Initialize the Isolation Forest model
  model = IsolationForest(contamination=0.01, random_state=42)
  
  # Fit the model to the data
  model.fit(cleaned_data)
  
  # Predict anomalies (1 for normal, -1 for anomaly)
  anomalies = model.predict(cleaned_data)
  
  # Add the anomaly column to the DataFrame
  cleaned_data['Anomaly'] = anomalies
  
  # Display the rows that are considered anomalies
  anomalies_df = cleaned_data[cleaned_data['Anomaly'] == -1]
  
  # Print the anomalies DataFrame
  anomalies_df.head()
  ```
8. **Counting and Visualizing Anomalies:** Count and visualize the number of anomalies detected by the Isolation Forest model.
  ```python
  # Count the number of anomalies
  anomaly_count = cleaned_data['Anomaly'].value_counts()
  
  # Print the count of anomalies
  print(anomaly_count)
  
  # Plot the count of anomalies
  plt.figure(figsize=(8, 6))
  anomaly_count.plot(kind='bar')
  plt.title('Count of Anomalies')
  plt.xlabel('Anomaly')
  plt.ylabel('Count')
  plt.xticks(ticks=[0, 1], labels=['Normal', 'Anomaly'], rotation=0)
  plt.show()
  
  # Plot anomalies in a scatter plot with different colors
  plt.figure(figsize=(10, 8))
  plt.scatter(cleaned_data[cleaned_data['Anomaly'] == 1]['AccessionYear'],
              cleaned_data[cleaned_data['Anomaly'] == 1]['Object Begin Date'],
              c='blue', label='Normal')
  plt.scatter(cleaned_data[cleaned_data['Anomaly'] == -1]['AccessionYear'],
              cleaned_data[cleaned_data['Anomaly'] == -1]['Object Begin Date'],
              c='red', label='Anomaly')
  plt.title('Scatter Plot of Anomalies')
  plt.xlabel('Accession Year')
  plt.ylabel('Object Begin Date')
  plt.legend()
  plt.show()
  ```
9. **Removing Anomalies and Plotting the New Graph:** Remove anomalies from the dataset and plot a scatter plot without these anomalies.
  ```python
  # Remove anomalies from the dataset
  cleaned_data_no_anomalies = cleaned_data[cleaned_data['Anomaly'] == 1]
  
  # Plot the new graph without anomalies
  plt.figure(figsize=(10, 8))
  plt.scatter(cleaned_data_no_anomalies['AccessionYear'],
              cleaned_data_no_anomalies['Object Begin Date'],
              c='blue', label='Normal')
  plt.title('Scatter Plot without Anomalies')
  plt.xlabel('Accession Year')
  plt.ylabel('Object Begin Date')
  plt.legend()
  plt.show()
  ```
10. **Filtering Outliers:** Filter rows in the dataset based on specific conditions.
  ```python
  # Filter rows where 'Object Begin Date' is greater than or equal to 'AccessionYear'
  outlier_rows_positive = cleaned_data[cleaned_data['Object Begin Date'] >= cleaned_data['AccessionYear']]
  outlier_rows_positive.shape
  
  # Filter rows where 'Object Begin Date' is less than or equal to -6000
  outlier_rows_negative = cleaned_data[cleaned_data['Object Begin Date'] <= -6000]
  outlier_rows_negative.shape
  
  # Filter rows with 'Object Begin Date' within a valid range
  cleaned_data = cleaned_data[cleaned_data['Object Begin Date'] <= 2025]
  cleaned_data = cleaned_data[cleaned_data['Object Begin Date'] >= -5000]
  cleaned_data.shape
  ```
11. **Reversing One-Hot Encoding**: Reverse one-hot encoding for the 'Department' column and drop the one-hot encoded columns.
    ```python
    # Identify one-hot encoded columns
    department_columns = [col for col in cleaned_data_no_anomalies.columns if col.startswith('Department_')]

    # Check if department_columns is not empty
    if department_columns:
        # Reverse one-hot encoding for the 'Department' column
        cleaned_data_no_anomalies.loc[:, 'Department'] = cleaned_data_no_anomalies[department_columns].idxmax(axis=1)
        cleaned_data_no_anomalies.loc[:, 'Department'] = cleaned_data_no_anomalies['Department'].apply(lambda x: x.split('_')[1])

        # Drop the one-hot encoded columns
        cleaned_data_no_anomalies = cleaned_data_no_anomalies.drop(columns=department_columns)
    else:
        print("No one-hot encoded columns found for 'Department_'")

    # Same For Non-Anomaly Data
    department_columns = [col for col in cleaned_data_no_one.columns if col.startswith('Department_')]

    # Check if department_columns is not empty
    if department_columns:
        # Reverse one-hot encoding for the 'Department' column
        cleaned_data_no_one.loc[:, 'Department'] = cleaned_data_no_one[department_columns].idxmax(axis=1)
        cleaned_data_no_one.loc[:, 'Department'] = cleaned_data_no_one['Department'].apply(lambda x: x.split('_')[1])

        # Drop the one-hot encoded columns
        cleaned_data_no_one = cleaned_data_no_one.drop(columns=department_columns)
    else:
        print("No one-hot encoded columns found for 'Department_'")
    ```

12. **Scatter Plot for Object Begin Date vs Accession Year by Department**: Create a scatter plot to visualize the relationship between object begin dates and accession years, grouped by department.
    ```python
    # Set the figure size for the plot
    plt.figure(figsize=(12, 6))

    # Loop through each unique department and create a scatter plot
    for department in cleaned_data_no_anomalies['Department'].unique():
        # Filter the data for the current department
        dept_data = cleaned_data_no_anomalies[cleaned_data_no_anomalies['Department'] == department]
        # Create a scatter plot for the current department
        plt.scatter(dept_data['Object Begin Date'], dept_data['AccessionYear'], label=department, alpha=0.5)

    # Add a title to the plot
    plt.title('Object Begin Date vs Accession Year by Department')

    # Label the X-axis
    plt.xlabel('Object Begin Date')

    # Label the Y-axis
    plt.ylabel('Accession Year')

    # Add a legend with the title 'Department' and position it outside the plot
    plt.legend(title='Department', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Display the plot
    plt.show()
    ```

13. **Data Visualization of Object Dates by Department**: Generate two plots: a scatter plot showing the mean of object begin dates by department and a boxplot showing the distribution of object begin dates by department.
    ```python
    # Calculate the mean Object Begin Date for each department
    obj_begin_means = cleaned_data_no_anomalies.groupby('Department')['Object Begin Date'].mean()

    # Create a figure with specified size
    plt.figure(figsize=(8, 6))  # Adjust figure size if needed

    # Generate a scatter plot of the mean accession years by department
    plt.scatter(obj_begin_means.index, obj_begin_means.values)

    # Add title and labels to the plot
    plt.title('Mean of Object Begin Date by Department')
    plt.xlabel('Departments')
    plt.ylabel('Object Begin Date')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=90)

    # Display the plot
    plt.show()

    # Boxplot of Object Dates by Department
    # Generate a boxplot of object begin dates by department using the default palette
    sns.boxplot(x=cleaned_data_no_anomalies['Department'], y=cleaned_data_no_anomalies['Object Begin Date'])

    # Set the theme for the plot with specified figure size
    sns.set_theme(rc={'figure.figsize': (20, 10)})

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=90)

    # Add title to the plot
    plt.title("Object Dates By Department")

    # Display the plot
    plt.show()
    ```

14. **Data Visualization of Accession Dates by Department**: Generate two plots: a scatter plot showing the mean of accession years by department and a boxplot showing the distribution of accession years by department.
    ```python
    # Calculate the mean accession year for each department
    acces_means = cleaned_data_no_anomalies.groupby('Department')['AccessionYear'].mean()

    # Create a figure with specified size
    plt.figure(figsize=(8, 6))  # Adjust figure size if needed

    # Generate a scatter plot of the mean accession years by department
    plt.scatter(acces_means.index, acces_means.values, color='green')

    # Add title and labels to the plot
    plt.title('Mean of Accession Years by Department')
    plt.xlabel('Departments')
    plt.ylabel('Accession Years')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=90)

    # Display the plot
    plt.show()

    # Boxplot of Accession Date By Department
    # Generate a boxplot of accession years by department using a green color palette
    sns.boxplot(x=cleaned_data_no_anomalies['Department'], y=cleaned_data_no_anomalies['AccessionYear'], color='green')

    # Set the theme for the plot with specified figure size
    sns.set_theme(rc={'figure.figsize': (20, 10)})

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=90)

    # Add title to the plot
    plt.title("Accession Year By Department")

    # Display the plot
    plt.show()
    ```

15. **Most Represented Department by Year**: Identify the most represented department for each year based on the 'Object Begin Date' and print the results.
    ```python
    # Group the data by 'Object Begin Date' and find the most common department for each year
    mcv_osd = cleaned_data_no_anomalies.groupby('Object Begin Date')['Department'].agg(lambda x: x.value_counts().index[0]).rename('Most Common Value')

    # Print the most common department for each year
    print(mcv_osd)
    ```

16. **Most Represented Department by Accession Year**: Identify the most represented department for each accession year based on the 'AccessionYear' and print the results.
    ```python
    # Group the data by 'AccessionYear' and find the most common department for each year
    mcv_acc = cleaned_data_no_anomalies.groupby('AccessionYear')['Department'].agg(lambda x: x.value_counts().index[0]).rename('Most Common Value')

    # Print the most common department for each accession year
    print(mcv_acc)
    ```

17. **Modeling Imports**: Import the necessary libraries and modules for building and evaluating machine learning models, specifically decision trees and random forests.
    ```python
    # Import metrics for evaluating model performance
    from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

    # Import module for splitting data into training and testing sets
    from sklearn.model_selection import train_test_split

    # Import decision tree classifier and plotting functions
    from sklearn.tree import DecisionTreeClassifier, plot_tree

    # Import random forest classifier
    from sklearn.ensemble import RandomForestClassifier

    # Import matplotlib for plotting
    import matplotlib.pyplot as plt

    # Import additional metrics and utilities
    from sklearn import metrics
    from sklearn.utils.multiclass import unique_labels
    from sklearn import tree

    # Import numpy for numerical operations
    import numpy as np
    ```
    18. **Data Splitting for Model Training and Testing**: Split the dataset into training and testing sets for both old and cleaned data. The target variable is 'Department', and some irrelevant columns are dropped before splitting.
    ```python
    # Splits With Old Data
    # Drop irrelevant columns from the old dataset
    X_old = cleaned_data_no_one.drop(['Department', 'Dimensions', 'Artist Begin Date', 'Artist End Date'], axis=1)

    # Sort the old dataset by 'Department'
    cleaned_data_no_one = cleaned_data_no_one.sort_values(by='Department')

    # Set the target variable for the old dataset
    y_old = cleaned_data_no_one.Department

    # Split the old dataset into training and testing sets (70% train, 30% test)
    X_train_old, X_test_old, y_train_old, y_test_old = train_test_split(X_old, y_old, test_size=0.3, random_state=1)

    # Splits With Cleaned Data
    # Drop irrelevant columns from the cleaned dataset
    X = cleaned_data_no_anomalies.drop(['Department', 'Dimensions', 'Artist Begin Date', 'Artist End Date'], axis=1)

    # Sort the cleaned dataset by 'Department'
    cleaned_data_no_anomalies = cleaned_data_no_anomalies.sort_values(by='Department')

    # Set the target variable for the cleaned dataset
    y = cleaned_data_no_anomalies.Department

    # Split the cleaned dataset into training and testing sets (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    ```

19. **Decision Tree Classifier on Old and New Data**: Train and predict using a Decision Tree Classifier on both old and cleaned datasets.
    ```python
    # Initialize the Decision Tree Classifier
    clf = DecisionTreeClassifier()

    # Train the classifier on the old training data
    clf = clf.fit(X_train_old, y_train_old)

    # Predict the target variable for the old test data
    y_pred_old = clf.predict(X_test_old)

    # Decision Tree on New Data
    # Initialize the Decision Tree Classifier
    clf = DecisionTreeClassifier()

    # Train the classifier on the cleaned training data
    clf = clf.fit(X_train, y_train)

    # Predict the target variable for the cleaned test data
    y_pred = clf.predict(X_test)
    ```

20. **Decision Tree Accuracy Evaluation**: Evaluate the accuracy of the Decision Tree Classifier on both old and cleaned datasets and print the results.
    ```python
    # Calculate and print the accuracy of the Decision Tree Classifier on the old test data
    print("Old Data Accuracy:", metrics.accuracy_score(y_test_old, y_pred_old))

    # Calculate and print the accuracy of the Decision Tree Classifier on the cleaned test data
    print("Cleaned Accuracy:", metrics.accuracy_score(y_test, y_pred))
    ```

21. **Plotting Confusion Matrix for Decision Tree Model**: Generate a confusion matrix for the Decision Tree model's predictions and visualize it using a heatmap.
    ```python
    # Get the unique labels from the test and predicted data
    labels = unique_labels(y_test, y_pred)

    # Compute the confusion matrix
    matrix = confusion_matrix(y_test, y_pred, labels=labels)

    # Normalize the confusion matrix, adding a small value to avoid division by zero
    epsilon = 1e-10
    matrix = matrix.astype('float') / (matrix.sum(axis=1)[:, np.newaxis] + epsilon)

    # Create a figure with specified size
    plt.figure(figsize=(16, 7))

    # Generate a heatmap for the confusion matrix
    sns.heatmap(matrix, annot=True, annot_kws={'size': 10}, cmap=plt.cm.Blues)

    # Set tick marks for x and y axes
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks + 0.5, labels, rotation=90)
    plt.yticks(tick_marks + 0.5, labels, rotation=0)

    # Add labels and title to the plot
    plt.xlabel('Prediction')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix for Decision Model')

    # Display the plot
    plt.show()
    ```

22. **Classification Report for Decision Tree Model**: Generate and print a classification report for the Decision Tree model's predictions, providing detailed metrics on the model's performance.
    ```python
    # Generate the classification report for the Decision Tree model's predictions
    class_report = classification_report(y_test, y_pred, zero_division=0)

    # Print the classification report
    print(f"Classification Report:\n{class_report}")
    ```

23. **Random Forest Classifier on Old and New Data**: Train and predict using a Random Forest Classifier on both old and cleaned datasets.
    ```python
    # Random Forest on Old Data
    # Initialize the Random Forest Classifier
    rf = RandomForestClassifier()

    # Train the classifier on the old training data
    rf = rf.fit(X_train_old, y_train_old)

    # Predict the target variable for the old test data
    y_pred_old = rf.predict(X_test_old)

    # Random Forest on New Data
    # Initialize the Random Forest Classifier
    rf = RandomForestClassifier()

    # Train the classifier on the cleaned training data
    rf = rf.fit(X_train, y_train)

    # Predict the target variable for the cleaned test data
    y_pred = rf.predict(X_test)
    ```

24. **Random Forest Accuracy Evaluation**: Evaluate the accuracy of the Random Forest Classifier on both old and cleaned datasets and print the results.
    ```python
    # Calculate and print the accuracy of the Random Forest Classifier on the old test data
    print("Old Data Accuracy:", metrics.accuracy_score(y_test_old, y_pred_old))

    # Calculate and print the accuracy of the Random Forest Classifier on the cleaned test data
    print("Cleaned Accuracy:", metrics.accuracy_score(y_test, y_pred))
    ```

25. **Plotting Confusion Matrix for Random Forest Model**: Generate a confusion matrix for the Random Forest model's predictions and visualize it using a heatmap.
    ```python
    # Get the unique labels from the test and predicted data
    labels = unique_labels(y_test, y_pred)

    # Compute the confusion matrix
    matrix = confusion_matrix(y_test, y_pred, labels=labels)

    # Normalize the confusion matrix, adding a small value to avoid division by zero
    epsilon = 1e-10
    matrix = matrix.astype('float') / (matrix.sum(axis=1)[:, np.newaxis] + epsilon)

    # Create a figure with specified size
    plt.figure(figsize=(16, 7))

    # Generate a heatmap for the confusion matrix
    sns.heatmap(matrix, annot=True, annot_kws={'size': 10}, cmap=plt.cm.Blues)

    # Set tick marks for x and y axes
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks + 0.5, labels, rotation=90)  # Shift x-axis labels to the right
    plt.yticks(tick_marks + 0.5, labels, rotation=0)  # Shift y-axis labels down

    # Add labels and title to the plot
    plt.xlabel('Prediction')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix for Random Forest Model')

    # Display the plot
    plt.show()
    ```
    26. **Classification Report for Random Forest Model**: Generate and print a classification report for the Random Forest model's predictions, providing detailed metrics on the model's performance.
    ```python
    # Generate the classification report for the Random Forest model's predictions
    class_report = classification_report(y_test, y_pred, zero_division=0)

    # Print the classification report
    print(f"Classification Report:\n{class_report}")
    ```
