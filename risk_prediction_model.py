import os
import json
import logging
from typing import Tuple, Dict, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, precision_recall_curve)
from sklearn.preprocessing import MultiLabelBinarizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class ESGRiskModelTrainer:
    def __init__(self, processed_data_dir: str = 'processed_esg_data',model_output_dir: str = 'esg_risk_models'):
        """
        Initialize ESG Risk Model Trainer
        
        Args:
            processed_data_dir: Directory containing processed ESG data
            model_output_dir: Directory to save trained models
        """
        self.processed_data_dir = processed_data_dir
        os.makedirs(model_output_dir, exist_ok=True)
        self.model_output_dir = model_output_dir
        
        # Risk categories
        self.risk_categories = ['environmental_risk', 'social_risk', 'governance_risk']
    
    def load_processed_data(self) -> pd.DataFrame:
        """
        Load processed ESG data
        
        Returns:
            Consolidated DataFrame of processed ESG data
        """
        consolidated_data = []
        for filename in os.listdir(self.processed_data_dir):
            if filename.endswith('.csv'):
                filepath = os.path.join(self.processed_data_dir, filename)
                try:
                    df = pd.read_csv(filepath)
                    consolidated_data.append(df)
                except Exception as e:
                    logger.error(f"Error loading {filename}: {e}")
        if not consolidated_data:
            raise ValueError("No processed data found")
        return pd.concat(consolidated_data, ignore_index=True)
    
    def prepare_model_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for model training
        
        Args:
            df: Input DataFrame
        
        Returns:
            Tuple of (features, labels)
        """
        # Select feature columns (exclude categorical and risk label columns)
        feature_columns = [col for col in df.columns if col.startswith('text_feature_') or col in ['text_length', 'keywords_count', 'is_eu_regulation', 'is_us_regulation']]
        # Prepare features
        X = df[feature_columns].values
        # Prepare multi-label risk targets
        y = df[self.risk_categories].values
        return X, y
    
    def build_deep_risk_model(self, input_shape: int) -> tf.keras.Model:
        """
        Build a deep neural network for multi-label ESG risk classification
        
        Args:
            input_shape: Number of input features
        
        Returns:
            Compiled Keras model
        """
        model = Sequential([
            # Input layer with L2 regularization
            Dense(64, activation='relu', input_shape=(input_shape,),kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            BatchNormalization(),
            Dropout(0.3),
            # Hidden layers
            Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            BatchNormalization(),
            Dropout(0.2),
            Dense(16, activation='relu'),
            BatchNormalization(),
            # Output layer for multi-label classification
            Dense(len(self.risk_categories), activation='sigmoid')
        ])
        
        # Compile model with focal loss for imbalanced data
        model.compile(optimizer=Adam(learning_rate=0.001),loss='binary_crossentropy',metrics=['accuracy'])
        return model
    
    def train_model(self) -> Dict[str, Any]:
        """
        Train ESG risk prediction model
        
        Returns:
            Dictionary of model training results
        """
        # Load and prepare data
        df = self.load_processed_data()
        X, y = self.prepare_model_data(df)
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Build model
        model = self.build_deep_risk_model(input_shape=X.shape[1])
        # Training callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
        # Train model
        history = model.fit(X_train, y_train,validation_split=0.2,epochs=100,batch_size=32,callbacks=[early_stopping, lr_reducer],verbose=1)
        # Evaluate model
        y_pred = model.predict(X_test)
        y_pred_binary = (y_pred > 0.5).astype(int)
        # Compute metrics
        metrics = {'classification_report': classification_report(y_test, y_pred_binary, target_names=self.risk_categories),
                   'roc_auc_scores': {category: roc_auc_score(y_test[:, i], y_pred[:, i]) for i, category in enumerate(self.risk_categories)}
        }
        # Save model
        model_path = os.path.join(self.model_output_dir, f'esg_risk_model_{pd.Timestamp.now().strftime("%Y%m%d")}')
        model.save(model_path)
        # Visualize results
        self._plot_training_history(history)
        self._plot_confusion_matrix(y_test, y_pred_binary)
        return {'model_path': model_path,'metrics': metrics,'history': history.history}
    
    def _plot_training_history(self, history):
        """
        Plot model training history
        
        Args:
            history: Keras training history
        """
        plt.figure(figsize=(12, 4))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_output_dir, 'training_history.png'))
        plt.close()
    
    def _plot_confusion_matrix(self, y_true, y_pred):
        """
        Plot confusion matrix for each risk category
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        """
        plt.figure(figsize=(15, 5))
        for i, category in enumerate(self.risk_categories):
            plt.subplot(1, 3, i+1)
            cm = confusion_matrix(y_true[:, i], y_pred[:, i])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {category}')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_output_dir, 'confusion_matrices.png'))
        plt.close()

# Main execution
def main():
    # Initialize and train model
    trainer = ESGRiskModelTrainer()
    try:
        results = trainer.train_model()
        # Log results
        logger.info("Model Training Completed")
        logger.info(f"Model saved at: {results['model_path']}")
        logger.info("Classification Report:")
        print(results['metrics']['classification_report'])
        # Save detailed metrics
        with open(os.path.join(trainer.model_output_dir, 'model_metrics.json'), 'w') as f:
            json.dump(results['metrics'], f, indent=2)
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        
if __name__ == "__main__":
    main()