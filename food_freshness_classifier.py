import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
)
from sklearn.pipeline import Pipeline
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import time
import pickle
from tensorflow.keras.models import load_model

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define the data directory
data_dir = r"C:\Users\ASUS\OneDrive\Desktop\AI\gan_output"
# Define output directory for all results
output_dir = r"C:\Users\ASUS\OneDrive\Desktop\AI\sensor_output"

# Function to load and combine datasets
def load_datasets(data_dir):
    """Load and combine different datasets for comparison"""
    datasets = {}

    # 1. Load scientific fruit dataset - most comprehensive
    scientific_path = os.path.join(data_dir, "enhanced_scientific_spoilage_dataset.csv")
    if os.path.exists(scientific_path):
        scientific_df = pd.read_csv(scientific_path)
        # Use only core sensor features for consistency
        core_features = ['Ambient_Temp', 'Humidity', 'Gas', 'VOC', 'CO', 'Core_Temp', 'Light', 'pH']
        target = 'Freshness'
        if all(feature in scientific_df.columns for feature in core_features) and target in scientific_df.columns:
            datasets['scientific'] = scientific_df[core_features + [target]]
            print(f"Loaded scientific dataset with {len(datasets['scientific'])} records")
    
    # 2. Load GAN-distribution matched dataset
    gan_matched_path = os.path.join(data_dir, "distribution_matched_gan_data.csv")
    if os.path.exists(gan_matched_path):
        gan_df = pd.read_csv(gan_matched_path)
        if 'Freshness' in gan_df.columns:
            core_features = ['Ambient_Temp', 'Humidity', 'Gas', 'VOC', 'CO', 'Core_Temp', 'Light', 'pH']
            available_features = [f for f in core_features if f in gan_df.columns]
            datasets['gan_matched'] = gan_df[available_features + ['Freshness']]
            print(f"Loaded GAN-matched dataset with {len(datasets['gan_matched'])} records")
    
    # 3. Load combined synthetic dataset
    synthetic_path = os.path.join(data_dir, "synthetic_food_freshness_data.csv")
    if os.path.exists(synthetic_path):
        synthetic_df = pd.read_csv(synthetic_path)
        if 'freshness' in synthetic_df.columns:
            # Remove rows with NaN in freshness column
            synthetic_df = synthetic_df.dropna(subset=['freshness'])
            datasets['synthetic'] = synthetic_df
            # Convert column names to match for consistency
            datasets['synthetic'] = datasets['synthetic'].rename(columns={
                'ambient_temp': 'Ambient_Temp',
                'humidity': 'Humidity',
                'gas': 'Gas',
                'voc': 'VOC',
                'co': 'CO',
                'core_temp': 'Core_Temp',
                'light': 'Light',
                'ph': 'pH',
                'freshness': 'Freshness'
            })
            print(f"Loaded synthetic dataset with {len(datasets['synthetic'])} records")
    
    # If scientific dataset exists, create a balanced subset for fair comparison
    if 'scientific' in datasets:
        balanced_df = datasets['scientific'].copy()
        # Ensure class balance
        min_class_count = balanced_df['Freshness'].value_counts().min()
        fresh_samples = balanced_df[balanced_df['Freshness'] == 'fresh'].sample(min_class_count)
        spoiled_samples = balanced_df[balanced_df['Freshness'] == 'spoiled'].sample(min_class_count)
        datasets['balanced'] = pd.concat([fresh_samples, spoiled_samples])
        print(f"Created balanced dataset with {len(datasets['balanced'])} records")
    
    # Create a combined dataset for unified model
    combined_datasets = []
    for name, df in datasets.items():
        df_copy = df.copy()
        df_copy['Source'] = name
        combined_datasets.append(df_copy)
    
    if combined_datasets:
        datasets['unified'] = pd.concat(combined_datasets, ignore_index=True)
        # Drop rows with NaN values in the freshness column
        datasets['unified'] = datasets['unified'].dropna(subset=['Freshness'])
        print(f"Created unified dataset with {len(datasets['unified'])} records from all sources")
    
    return datasets

# Preprocess datasets for modeling
def preprocess_data(df, test_size=0.2):
    """Preprocess a dataframe and split into train/test sets"""
    # Standard feature columns (consistent across datasets)
    # Explicitly exclude non-numeric columns like 'Source'
    exclude_cols = ['Freshness', 'freshness', 'Source']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Get target column
    target_col = 'Freshness' if 'Freshness' in df.columns else 'freshness'
    
    # Convert target to binary
    le = LabelEncoder()
    y = le.fit_transform(df[target_col])
    class_names = le.classes_
    print(f"Classes: {class_names}, encoded as {np.unique(y)}")
    
    # Get features
    X = df[feature_cols].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    # Standard scaling (fit only on training data to avoid data leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return {
        'X_train': X_train,
        'X_test': X_test, 
        'y_train': y_train,
        'y_test': y_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'feature_names': feature_cols,
        'class_names': class_names,
        'scaler': scaler
    }

# Model building functions
def build_classical_models():
    """Create a dict of classical ML models"""
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }
    return models

def build_neural_network(input_dim):
    """Build a neural network for binary classification"""
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(16, activation='relu'),
        BatchNormalization(),
        
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

# Enhanced evaluation function for classical models
def evaluate_model(model, preprocessed_data, model_name):
    """Train and evaluate a classical ML model with enhanced metrics"""
    # Get data
    X_train = preprocessed_data['X_train_scaled']
    X_test = preprocessed_data['X_test_scaled']
    y_train = preprocessed_data['y_train']
    y_test = preprocessed_data['y_test']
    class_names = preprocessed_data['class_names']
    
    # Train and time the model
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Generate detailed classification report
    class_report = classification_report(y_test, y_pred, target_names=class_names)
    
    # Calculate cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    
    # Compute ROC curve and AUC if probabilities available
    roc_auc = None
    if y_pred_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
    
    # Return results as dict
    results = {
        'name': model_name,
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'classification_report_text': class_report,
        'cv_scores': cv_scores,
        'cv_mean': np.mean(cv_scores),
        'cv_std': np.std(cv_scores),
        'training_time': training_time,
        'roc_auc': roc_auc
    }
    
    if y_pred_proba is not None:
        results['fpr'] = fpr
        results['tpr'] = tpr
    
    print(f"{model_name}: Accuracy={accuracy:.4f}, F1={f1:.4f}, CV={np.mean(cv_scores):.4f}±{np.std(cv_scores):.4f}")
    return results

# Enhanced neural network evaluation
def evaluate_nn(input_dim, preprocessed_data, model_name="Neural Network"):
    """Train and evaluate a neural network model with enhanced metrics"""
    # Get data
    X_train = preprocessed_data['X_train_scaled']
    X_test = preprocessed_data['X_test_scaled']
    y_train = preprocessed_data['y_train']
    y_test = preprocessed_data['y_test']
    class_names = preprocessed_data['class_names']
    
    # Build model
    model = build_neural_network(input_dim)
    
    # Early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    
    # Train the model and time it
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=0
    )
    training_time = time.time() - start_time
    
    # Predict and evaluate
    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Generate detailed classification report
    class_report = classification_report(y_test, y_pred, target_names=class_names)
    
    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Learning curves plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Learning Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'nn_learning_curves.png'))
    plt.close()
    
    # Return results
    results = {
        'name': model_name,
        'model': model,
        'history': history,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'classification_report_text': class_report,
        'training_time': training_time,
        'roc_auc': roc_auc,
        'fpr': fpr,
        'tpr': tpr
    }
    
    print(f"{model_name}: Accuracy={accuracy:.4f}, F1={f1:.4f}")
    return results

# Enhanced function to visualize model comparison
def visualize_model_comparison(all_results, dataset_name):
    """Create visualizations comparing model performance"""
    # Extract metrics for comparison
    model_names = [result['name'] for result in all_results]
    accuracies = [result['accuracy'] for result in all_results]
    f1_scores = [result['f1_score'] for result in all_results]
    precisions = [result['precision'] for result in all_results]
    recalls = [result['recall'] for result in all_results]
    training_times = [result['training_time'] for result in all_results]
    roc_aucs = [result.get('roc_auc', 0) for result in all_results]
    
    # Create figure for metrics comparison
    plt.figure(figsize=(15, 10))
    
    # Plot accuracy, precision, recall, F1
    plt.subplot(2, 2, 1)
    metrics_df = pd.DataFrame({
        'Accuracy': accuracies,
        'Precision': precisions,
        'Recall': recalls,
        'F1 Score': f1_scores,
        'ROC AUC': roc_aucs
    }, index=model_names)
    
    metrics_df.plot(kind='bar', ax=plt.gca())
    plt.title(f'Model Performance Metrics: {dataset_name}')
    plt.ylabel('Score')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    
    # Plot training times
    plt.subplot(2, 2, 2)
    plt.bar(model_names, training_times, color='skyblue')
    plt.title('Training Time Comparison')
    plt.ylabel('Time (seconds)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    
    # Plot ROC curves (if available)
    plt.subplot(2, 2, 3)
    for result in all_results:
        if 'fpr' in result and 'tpr' in result:
            plt.plot(result['fpr'], result['tpr'], label=f"{result['name']} (AUC = {result['roc_auc']:.4f})")
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    
    # Save confusion matrices for each model
    plt.figure(figsize=(15, 10))
    nrows = int(np.ceil(len(all_results) / 2))
    for i, result in enumerate(all_results):
        plt.subplot(nrows, 2, i+1)
        cm = result['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(f'{result["name"]} Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'all_confusion_matrices_{dataset_name}.png'))
    plt.close()
    
    # Save classification reports as text files
    class_report_path = os.path.join(output_dir, f'classification_reports_{dataset_name}.txt')
    with open(class_report_path, 'w') as f:
        for result in all_results:
            f.write(f"\n{'='*50}\n")
            f.write(f"Model: {result['name']}\n")
            f.write(f"{'='*50}\n")
            if 'classification_report_text' in result:
                f.write(result['classification_report_text'])
            f.write(f"\nAccuracy: {result['accuracy']:.4f}\n")
            f.write(f"F1 Score: {result['f1_score']:.4f}\n")
            if 'cv_mean' in result:
                f.write(f"CV Score: {result['cv_mean']:.4f} ± {result['cv_std']:.4f}\n")
            f.write("\n\n")
    
    # Additional plot for model accuracy comparison across datasets
    return metrics_df

# Function to build a unified model using the best approach
def build_unified_model(datasets):
    """Build a unified model using all datasets"""
    print("\n" + "="*70)
    print("Building Unified Food Freshness Prediction Model")
    print("="*70)
    
    # Get unified dataset
    if 'unified' not in datasets:
        print("Unified dataset not available. Using scientific dataset as fallback.")
        df = datasets.get('scientific', next(iter(datasets.values())))
    else:
        df = datasets['unified']
    
    print(f"Training unified model on {len(df)} samples")
    
    # Preprocess data
    preprocessed = preprocess_data(df)
    
    # Try all models to find the best one
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_leaf=1, random_state=42),
        'XGBoost': xgb.XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
    }
    
    print("Training and evaluating multiple algorithms...")
    
    # Track results
    results = []
    best_model = None
    best_score = 0
    best_model_name = ""
    
    # Evaluate each model
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        result = evaluate_model(model, preprocessed, name)
        results.append(result)
        
        # Check if this is the best model
        if result['f1_score'] > best_score:
            best_score = result['f1_score']
            best_model = model
            best_model_name = name
    
    # Also evaluate neural network
    print("\nEvaluating Neural Network...")
    input_dim = preprocessed['X_train_scaled'].shape[1]
    nn_result = evaluate_nn(input_dim, preprocessed, "Neural Network")
    results.append(nn_result)
    
    # Check if neural network is the best
    if nn_result['f1_score'] > best_score:
        best_score = nn_result['f1_score']
        best_model = nn_result['model']
        best_model_name = "Neural Network"
    
    # Create visualizations
    print("\nCreating unified model visualizations...")
    metrics_df = visualize_model_comparison(results, "unified_model")
    
    print(f"\nBest model: {best_model_name} with F1 score: {best_score:.4f}")
    
    # Save the best model
    if best_model_name == "Neural Network":
        best_model.save(os.path.join(output_dir, "best_unified_model.h5"))
    else:
        # Save sklearn model using pickle
        with open(os.path.join(output_dir, "best_unified_model.pkl"), "wb") as f:
            pickle.dump(best_model, f)
    
    print(f"Best unified model saved to {output_dir}/best_unified_model.h5/pkl")
    
    # Generate and save detailed model information
    with open(os.path.join(output_dir, "unified_model_details.txt"), "w") as f:
        f.write("="*50 + "\n")
        f.write("UNIFIED FOOD FRESHNESS PREDICTION MODEL\n")
        f.write("="*50 + "\n\n")
        f.write(f"Best model: {best_model_name}\n")
        f.write(f"F1 Score: {best_score:.4f}\n\n")
        f.write("Model performance comparison:\n\n")
        
        # Write performance summary for all models
        model_summary = metrics_df.sort_values('F1 Score', ascending=False)
        f.write(model_summary.to_string() + "\n\n")
        
        # Write dataset information
        f.write("Dataset information:\n")
        f.write(f"Total samples: {len(df)}\n")
        value_counts = df['Freshness'].value_counts()
        for value, count in value_counts.items():
            f.write(f"  - {value}: {count} samples ({count/len(df)*100:.2f}%)\n")
        
        # Write feature information
        f.write("\nTop features by importance:\n")
        feature_importance = analyze_feature_importance("unified")
        for feature, importance in zip(feature_importance['feature_names'], feature_importance['importances']):
            f.write(f"  - {feature}: {importance:.4f}\n")
    
    # Print path to detailed info
    print(f"Detailed model information saved to {output_dir}/unified_model_details.txt")
    
    return {
        'model': best_model,
        'model_name': best_model_name,
        'score': best_score,
        'results': results
    }

# Function to run the complete modeling workflow
def run_modeling_workflow():
    """Run the entire modeling workflow"""
    # Load all datasets
    print("Loading datasets...")
    datasets = load_datasets(data_dir)
    
    if len(datasets) == 0:
        print("No valid datasets found in the specified directory.")
        return
    
    # Store results across datasets for comparison
    all_dataset_results = {}
    comparison_metrics = []
    
    # First process each dataset individually
    for dataset_name, df in datasets.items():
        # Skip unified dataset for individual processing
        if dataset_name == "unified":
            continue
            
        print(f"\n\n{'='*50}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*50}")
        
        try:
            # Preprocess data
            print(f"Preprocessing {dataset_name} dataset...")
            preprocessed = preprocess_data(df)
            
            # Build classical models
            print("Building and evaluating classical models...")
            models = build_classical_models()
            
            # Evaluate classical models
            results = []
            for name, model in models.items():
                model_results = evaluate_model(model, preprocessed, name)
                results.append(model_results)
            
            # Build and evaluate neural network
            print("Building and evaluating neural network...")
            input_dim = preprocessed['X_train_scaled'].shape[1]
            nn_results = evaluate_nn(input_dim, preprocessed)
            results.append(nn_results)
            
            # Store all results for this dataset
            all_dataset_results[dataset_name] = results
            
            # Visualize model comparison for this dataset
            print(f"Creating visualizations for {dataset_name}...")
            metrics_df = visualize_model_comparison(results, dataset_name)
            metrics_df['Dataset'] = dataset_name
            comparison_metrics.append(metrics_df)
            
        except Exception as e:
            print(f"Error processing dataset {dataset_name}: {str(e)}")
    
    # Create cross-dataset comparison
    if comparison_metrics:
        try:
            cross_dataset_comparison = pd.concat(comparison_metrics)
            
            # Plot cross-dataset comparison
            plt.figure(figsize=(15, 10))
            
            # Set up the pivot for easier plotting
            pivot_df = cross_dataset_comparison.reset_index()
            pivot_df = pivot_df.melt(id_vars=['index', 'Dataset'], 
                                     var_name='Metric', 
                                     value_name='Score')
            
            # Plot
            g = sns.catplot(
                data=pivot_df, 
                x='index', y='Score', 
                hue='Dataset', col='Metric',
                kind='bar', height=4, aspect=1.2, 
                col_wrap=3, sharex=False
            )
            
            g.set_axis_labels("Model", "Score")
            g.set_xticklabels(rotation=45)
            g.fig.suptitle('Cross-Dataset Model Performance Comparison', fontsize=16)
            g.fig.subplots_adjust(top=0.9)
            
            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'cross_dataset_comparison.png'))
            
            # Create overall summary table
            summary = cross_dataset_comparison.groupby('Dataset').mean().sort_values('F1 Score', ascending=False)
            summary.to_csv(os.path.join(output_dir, 'model_performance_summary.csv'))
            print("\nOverall performance summary:")
            print(summary)
            
        except Exception as e:
            print(f"Error creating cross-dataset comparison: {str(e)}")
    
    # Now build the unified model that can predict both classes
    unified_model_results = build_unified_model(datasets)
    
    print("\nModeling workflow complete. Results saved to the sensor_output directory.")
    
    return unified_model_results

# Function to perform feature importance analysis
def analyze_feature_importance(dataset_name='scientific'):
    """Analyze feature importance using Random Forest"""
    # Load a dataset (preferably the scientific one)
    datasets = load_datasets(data_dir)
    
    if dataset_name not in datasets:
        print(f"Dataset {dataset_name} not found. Available datasets: {list(datasets.keys())}")
        if len(datasets) > 0:
            dataset_name = list(datasets.keys())[0]
            print(f"Using {dataset_name} instead.")
        else:
            print("No datasets available.")
            return
    
    df = datasets[dataset_name]
    
    # Preprocess
    preprocessed = preprocess_data(df)
    
    # Train a Random Forest model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(preprocessed['X_train_scaled'], preprocessed['y_train'])
    
    # Get feature importances
    importances = rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
    
    # Sort importances
    indices = np.argsort(importances)[::-1]
    feature_names = preprocessed['feature_names']
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.title(f'Feature Importance: {dataset_name} dataset')
    plt.bar(range(len(importances)), importances[indices], color="b", yerr=std[indices], align="center")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.xlim([-1, len(importances)])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'feature_importance_{dataset_name}.png'))
    
    # Print feature importance ranking
    print("\nFeature importance ranking:")
    for i in indices:
        print(f"{feature_names[i]}: {importances[i]:.4f}")
        
    return {
        'feature_names': [feature_names[i] for i in indices],
        'importances': importances[indices]
    }

# Function to make predictions with the unified model
def predict_freshness(sensors_data):
    """Make predictions using the best unified model"""
    # Load the best model
    model_path = os.path.join(output_dir, "best_unified_model.pkl")
    h5_model_path = os.path.join(output_dir, "best_unified_model.h5")
    
    model = None
    is_nn = False
    
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    elif os.path.exists(h5_model_path):
        model = load_model(h5_model_path)
        is_nn = True
    else:
        print("No saved model found. Please train the model first.")
        return None
    
    # Load the scaler from the saved file
    scaler_path = os.path.join(output_dir, "unified_scaler.pkl")
    if not os.path.exists(scaler_path):
        print("No scaler found. Please train the model first.")
        return None
    
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    
    # Ensure sensors_data is in the right format
    features = ['Ambient_Temp', 'Humidity', 'Gas', 'VOC', 'CO', 'Core_Temp', 'Light', 'pH']
    
    # Convert to numpy array if needed
    if isinstance(sensors_data, pd.DataFrame):
        # Ensure columns match expected features
        sensors_data = sensors_data[features].values
    
    # Scale the data
    scaled_data = scaler.transform(sensors_data)
    
    # Make prediction
    if is_nn:
        proba = model.predict(scaled_data, verbose=0)
        # Convert probabilities to classes
        prediction = (proba > 0.5).astype(int)
    else:
        prediction = model.predict(scaled_data)
        proba = model.predict_proba(scaled_data)
    
    # Map predictions back to classes
    class_mapping = {0: 'fresh', 1: 'spoiled'}
    result = [class_mapping[p] for p in prediction]
    
    return {
        'predictions': result,
        'probabilities': proba
    }

# Main function
if __name__ == "__main__":
    # Create output directories if they don't exist
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    print("Starting unified food freshness classification modeling...")
    unified_model = run_modeling_workflow()
    
    print("\nAnalyzing feature importance...")
    feature_importance = analyze_feature_importance("unified")
    
    # Save the scaler for future predictions
    datasets = load_datasets(data_dir)
    if 'unified' in datasets:
        preprocessed = preprocess_data(datasets['unified'])
        with open(os.path.join(output_dir, "unified_scaler.pkl"), "wb") as f:
            pickle.dump(preprocessed['scaler'], f)
    
    print("\nProcess completed successfully!")
