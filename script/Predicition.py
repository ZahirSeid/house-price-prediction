import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(train_path, test_path):
    """Load train and test datasets."""
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    return train_data, test_data

def preprocess_data(data, features):
    """Preprocess the data: select features, handle missing values."""
    X = data[features]
    X.fillna(X.mean(), inplace=True)  # Fill missing values with mean
    return X

def train_model(X_train, y_train):
    """Train a Random Forest Regressor model."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_val, y_val):
    """Evaluate the model performance on the validation set."""
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print(f"Validation RMSE: {rmse}")
    return rmse

def plot_feature_importance(model, features):
    """Plot the feature importance."""
    feature_importances = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importances)
    plt.title('Feature Importance')
    plt.show()

def save_submission(test_data, predictions, output_path):
    """Save the predictions in the required submission format."""
    submission = pd.DataFrame({
        'Id': test_data['Id'],
        'SalePrice': predictions
    })
    submission.to_csv(output_path, index=False)
    print("Submission file created successfully.")

if __name__ == "__main__":
    # File paths
    train_path = 'data/train.csv'
    test_path = 'data/test.csv'
    output_path = 'data/my_submission.csv'

    # Load datasets
    train_data, test_data = load_data(train_path, test_path)

    # Features and target selection
    features = ['GrLivArea', 'OverallQual', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
    X_train = preprocess_data(train_data, features)
    y_train = train_data['SalePrice']
    
    # Split data for validation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Evaluate the model
    evaluate_model(model, X_val, y_val)
    
    # Plot feature importance
    plot_feature_importance(model, features)

    # Preprocess the test set and make predictions
    X_test = preprocess_data(test_data, features)
    test_preds = model.predict(X_test)
    
    # Save the submission file
    save_submission(test_data, test_preds, output_path)
