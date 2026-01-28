import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import os


def train_and_evaluate_classifier(X_train, X_test, y_train, y_test, classifier_name='RandomForest'):
    """
    Train and evaluate a classifier.
    
    Parameters:
    - X_train, X_test: feature matrices
    - y_train, y_test: labels
    - classifier_name: 'RandomForest' or 'KNN'
    
    Returns:
    - trained model, predictions, accuracy, confusion matrix
    """
    if classifier_name == 'RandomForest':
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    elif classifier_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=5)
    else:
        raise ValueError(f'Unknown classifier: {classifier_name}')
    
    # Train
    print(f'Training {classifier_name}...')
    clf.fit(X_train, y_train)
    
    # Predict
    y_pred = clf.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f'  ✓ {classifier_name} trained')
    print(f'  - Accuracy: {accuracy:.4f}')
    
    return clf, y_pred, accuracy, cm


def plot_confusion_matrix(cm, labels, classifier_name, output_path=None):
    """
    Plot confusion matrix heatmap.
    
    Parameters:
    - cm: confusion matrix
    - labels: activity labels
    - classifier_name: name of the classifier
    - output_path: path to save the figure
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels, cbar=True)
    plt.title(f'Confusion Matrix - {classifier_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=100)
        print(f'  ✓ Confusion matrix saved to {output_path}')
    
    return plt.gcf()


def main():
    """
    Main pipeline: load features, train and evaluate classifiers.
    """
    # Load features
    print('Loading features...')
    features_path = 'data/features.csv'
    
    if not os.path.exists(features_path):
        print(f'Error: {features_path} not found')
        print('Please run feature_engineering.py first')
        return
    
    df = pd.read_csv(features_path)
    print(f'  ✓ Loaded {len(df)} samples with {len(df.columns) - 1} features')
    
    # Prepare data
    X = df.drop('activity', axis=1)
    y = df['activity']
    
    # Handle NaN values
    X = X.fillna(0)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f'\nData split: {len(X_train)} train, {len(X_test)} test')
    
    # Standardize features
    print('Standardizing features...')
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print('  ✓ Features standardized')
    
    # Train and evaluate Random Forest
    print('\n' + '='*50)
    print('RANDOM FOREST CLASSIFIER')
    print('='*50)
    rf_model, rf_pred, rf_acc, rf_cm = train_and_evaluate_classifier(
        X_train_scaled, X_test_scaled, y_train, y_test, 'RandomForest'
    )
    
    # Print detailed report
    print('\nClassification Report:')
    print(classification_report(y_test, rf_pred))
    
    # Plot confusion matrix
    os.makedirs('results', exist_ok=True)
    plot_confusion_matrix(rf_cm, df['activity'].unique(), 'Random Forest', 
                         'results/confusion_matrix_rf.png')
    
    # Train and evaluate KNN
    print('\n' + '='*50)
    print('K-NEAREST NEIGHBORS CLASSIFIER')
    print('='*50)
    knn_model, knn_pred, knn_acc, knn_cm = train_and_evaluate_classifier(
        X_train_scaled, X_test_scaled, y_train, y_test, 'KNN'
    )
    
    # Print detailed report
    print('\nClassification Report:')
    print(classification_report(y_test, knn_pred))
    
    # Plot confusion matrix
    plot_confusion_matrix(knn_cm, df['activity'].unique(), 'K-Nearest Neighbors', 
                         'results/confusion_matrix_knn.png')
    
    # Comparison summary
    print('\n' + '='*50)
    print('COMPARISON SUMMARY')
    print('='*50)
    print(f'Random Forest Accuracy:      {rf_acc:.4f}')
    print(f'K-Nearest Neighbors Accuracy: {knn_acc:.4f}')
    print(f'Difference:                   {abs(rf_acc - knn_acc):.4f}')
    
    if rf_acc > knn_acc:
        print(f'\n→ Random Forest performs better (+{(rf_acc - knn_acc):.4f})')
    else:
        print(f'\n→ K-Nearest Neighbors performs better (+{(knn_acc - rf_acc):.4f})')
    
    plt.show()


if __name__ == '__main__':
    main()