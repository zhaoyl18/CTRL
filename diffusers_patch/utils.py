from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
import torch
import torch.nn as nn
import numpy as np

def compute_classification_metrics(pred_classes, true_classes):
    # Compute accuracy
    correct_predictions = (pred_classes == true_classes).sum()
    total_predictions = true_classes.shape[0]
    accuracy = correct_predictions / total_predictions

    # Convert tensors to numpy arrays for sklearn compatibility
    pred_classes_np = pred_classes.cpu().numpy()
    true_classes_np = true_classes.cpu().numpy()

    # Calculate precision, recall, and F1 score for each class
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_classes_np, 
        pred_classes_np, 
        average=None,
        zero_division=0
    )

    # Calculate macro versions of the metrics
    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1.mean()

    # Returning all the metrics in a dictionary
    metrics = {
        'accuracy': accuracy.item(),
        'precision': precision,
        'recall': recall,
        'F1 score': f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_F1': macro_f1,
    }
    return metrics

def compute_regression_metrics(pred_values, true_values):
    # Convert tensors to numpy arrays for sklearn compatibility
    pred_values_np = pred_values.cpu().numpy()
    true_values_np = true_values.cpu().numpy()

    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(true_values_np, pred_values_np)

    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(true_values_np, pred_values_np)

    # Calculate Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)

    # Calculate R-squared (R²)
    r2 = r2_score(true_values_np, pred_values_np)

    # Calculate Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((true_values_np - pred_values_np) / true_values_np)) * 100

    # Calculate Explained Variance Score
    explained_variance = explained_variance_score(true_values_np, pred_values_np)

    # Returning all the metrics in a dictionary
    metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape,
        'Explained Variance': explained_variance,
    }
    return metrics

class TemperatureScaler(nn.Module):
    def __init__(self):
        super(TemperatureScaler, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1))  # Initialize temperature

    def forward(self, logits):
        return logits / self.temperature

    def calibrate(self, logits, labels):
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            loss = nn.CrossEntropyLoss()(self.forward(logits), labels)
            optimizer.zero_grad()
            loss.backward()
            return loss

        optimizer.step(eval)