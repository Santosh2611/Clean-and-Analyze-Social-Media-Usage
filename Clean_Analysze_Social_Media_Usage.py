import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.stats import f_oneway
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def generate_random_data(n, categories):
    """
    Generate random data for the given number of records and categories.
    
    Args:
    n (int): Number of records to generate.
    categories (list): List of categories.
    
    Returns:
    pandas.DataFrame: Generated random data.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    
    # Create random data with Date, Category, and Likes
    data = {
        'Date': pd.date_range('2023-01-01', periods=n),
        'Category': [random.choice(categories) for _ in range(n)],
        'Likes': np.random.randint(0, 10000, size=n)
    }
    return pd.DataFrame(data)

def display_dataframe_info(df):
    """
    Display overview of the dataframe.
    
    Args:
    df (pandas.DataFrame): Input dataframe.
    """
    print("DataFrame Overview:")
    print(df.head())  # Display first few rows of the dataframe
    print("\nDataset Overview:")
    print(df.info())  # Display dataset information

def visualize_category_engagement(df):
    """
    Visualize average likes by category.
    
    Args:
    df (pandas.DataFrame): Input dataframe.
    """
    if not df.empty:
        # Calculate average likes for each category
        avg_likes_by_category = df.groupby('Category')['Likes'].mean().sort_values()
        plt.figure(figsize=(10, 6))  # Set figure size for the plot
        # Create a bar plot for average likes by category
        plt.bar(avg_likes_by_category.index, avg_likes_by_category.values, color='skyblue')
        plt.title('Average Likes by Category')  # Set title for the plot
        plt.xlabel('Category')  # Set label for x-axis
        plt.ylabel('Average Likes')  # Set label for y-axis
        plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
        plt.show()  # Display the plot

def perform_anova_test(df, categories):
    """
    Perform ANOVA test on the likes for different categories.
    
    Args:
    df (pandas.DataFrame): Input dataframe.
    categories (list): List of categories.
    """
    # Extract likes for each category
    anova_data = [df[df['Category'] == cat]['Likes'] for cat in categories]
    # Perform ANOVA test
    anova_result = f_oneway(*anova_data)
    print("\nANOVA Test Result:")
    print(anova_result)  # Display ANOVA test result

def build_logistic_regression_model(df):
    """
    Build and evaluate a logistic regression model.
    
    Args:
    df (pandas.DataFrame): Input dataframe.
    """
    if 'Likes' in df:
        X, y = df[['Likes']], (df['Likes'] > df['Likes'].mean()).astype(int)
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create and train a logistic regression model
        lr_model = LogisticRegression()
        lr_model.fit(X_train, y_train)
        # Make predictions using the trained model
        lr_predictions = lr_model.predict(X_test)
        
        print("\nLogistic Regression Model Accuracy:")
        print(accuracy_score(y_test, lr_predictions))  # Display model accuracy
        print("\nClassification Report:")
        print(classification_report(y_test, lr_predictions))  # Display classification report

# Main function to run the analysis
def main():
    n = 500
    categories = ['Food', 'Travel', 'Fashion', 'Fitness', 'Music', 'Culture', 'Family', 'Health']
    df = generate_random_data(n, categories)  # Generate random data
    display_dataframe_info(df)  # Display dataframe overview
    visualize_category_engagement(df)  # Visualize average likes by category
    perform_anova_test(df, categories)  # Perform ANOVA test
    build_logistic_regression_model(df)  # Build and evaluate a logistic regression model

if __name__ == "__main__":
    main()  # Execute the main function
