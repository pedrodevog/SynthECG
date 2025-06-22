import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

from model import xresnet1d101, ClassificationModel
from dataset import TimeseriesDatasetCrops, ToTensor
from utils import evaluate_experiment

class FastaiModel(ClassificationModel):
    def __init__(self, name, n_classes, freq, outputfolder, input_shape, input_size=2.5, input_channels=12,
                chunkify_train=False, chunkify_valid=True, bs=128, ps_head=0.5, lin_ftrs_head=[128],
                wd=1e-2, epochs=50, lr=1e-2, kernel_size=5, loss="binary_cross_entropy", 
                early_stopping=None, aggregate_fn="max"):
        super().__init__()
        
        self.name = name
        self.num_classes = n_classes 
        self.target_fs = freq
        self.outputfolder = Path(outputfolder)

        self.input_size = int(input_size * self.target_fs)
        self.input_channels = input_channels

        self.chunkify_train = chunkify_train
        self.chunkify_valid = chunkify_valid

        self.chunk_length_train = 2 * self.input_size
        self.chunk_length_valid = self.input_size

        self.min_chunk_length = self.input_size

        self.stride_length_train = self.input_size
        self.stride_length_valid = self.input_size // 2

        self.bs = bs
        self.ps_head = ps_head
        self.lin_ftrs_head = lin_ftrs_head
        self.wd = wd
        self.epochs = epochs
        self.lr = lr
        self.kernel_size = kernel_size
        self.loss = loss
        self.input_shape = input_shape
        self.early_stopping = early_stopping
        self.aggregate_fn = aggregate_fn
        
    def fit(self, X_train, y_train, X_val, y_val):
        # Convert everything to float32
        X_train = [l.astype(np.float32) for l in X_train]
        X_val = [l.astype(np.float32) for l in X_val]
        y_train = [l.astype(np.float32) for l in y_train]
        y_val = [l.astype(np.float32) for l in y_val]
        
        # Create DataFrames
        df_train = pd.DataFrame({"data": range(len(X_train)), "label": y_train})
        df_valid = pd.DataFrame({"data": range(len(X_val)), "label": y_val})
        
        # Set up transforms
        tfms_ptb_xl = [ToTensor()]
        
        # Create datasets
        ds_train = TimeseriesDatasetCrops(
            df_train, 
            self.input_size,
            num_classes=self.num_classes,
            chunk_length=self.chunk_length_train if self.chunkify_train else 0,
            min_chunk_length=self.min_chunk_length,
            stride=self.stride_length_train,
            transforms=tfms_ptb_xl,
            annotation=False,
            col_lbl="label",
            npy_data=X_train
        )
        
        ds_valid = TimeseriesDatasetCrops(
            df_valid,
            self.input_size,
            num_classes=self.num_classes,
            chunk_length=self.chunk_length_valid if self.chunkify_valid else 0,
            min_chunk_length=self.min_chunk_length,
            stride=self.stride_length_valid,
            transforms=tfms_ptb_xl,
            annotation=False,
            col_lbl="label",
            npy_data=X_val
        )
        
        # Create DataLoaders
        train_loader = DataLoader(ds_train, batch_size=self.bs, shuffle=True)
        val_loader = DataLoader(ds_valid, batch_size=self.bs)
        
        # Set up the model
        self.input_channels = self.input_shape[-1]
        self.model = xresnet1d101(
            num_classes=self.num_classes, 
            input_channels=self.input_channels, 
            kernel_size=self.kernel_size, 
            ps_head=self.ps_head, 
            lin_ftrs_head=self.lin_ftrs_head
        )
        
        # Set up the loss function
        if self.loss == "binary_cross_entropy":
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Set up the optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        
        # Move model to GPU if available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # Train the model
        best_val_loss = float('inf')
        
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")):
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(val_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    
                    val_loss += loss.item()
            
            # Calculate average losses
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save model if validation loss improves
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_model(self.model, self.outputfolder / f"{self.name}.pth")
                print(f"Model saved - validation loss improved to {best_val_loss:.4f}")
        
        # Load the best model
        self._load_model(self.outputfolder / f"{self.name}.pth")
    
    def predict(self, X):
        X = [l.astype(np.float32) for l in X]
        y_dummy = [np.ones(self.num_classes, dtype=np.float32) for _ in range(len(X))]
        
        df = pd.DataFrame({"data": range(len(X)), "label": y_dummy})
        
        ds = TimeseriesDatasetCrops(
            df,
            self.input_size,
            num_classes=self.num_classes,
            chunk_length=self.chunk_length_valid,
            min_chunk_length=self.min_chunk_length,
            stride=self.stride_length_valid,
            transforms=[ToTensor()],
            annotation=False,
            col_lbl="label",
            npy_data=X
        )
        
        loader = DataLoader(ds, batch_size=self.bs)
        
        # Evaluation mode
        self.model.eval()
        
        # Get predictions
        all_preds = []
        id_mapping = ds.get_id_mapping()
        
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(loader):
                data = data.to(self.device)
                output = self.model(data)
                preds = torch.sigmoid(output).detach().cpu().numpy()
                all_preds.extend(preds)
        
        all_preds = np.array(all_preds)
        
        # Aggregate predictions for each sample
        return self._aggregate_predictions(all_preds, idmap=id_mapping)
    
    def _aggregate_predictions(self, preds, idmap):
        if len(idmap) != len(np.unique(idmap)):
            preds_aggregated = []
            
            for i in np.unique(idmap):
                preds_local = preds[np.where(idmap == i)[0]]
                if self.aggregate_fn == "mean":
                    preds_aggregated.append(np.mean(preds_local, axis=0))
                else:  # max
                    preds_aggregated.append(np.amax(preds_local, axis=0))
            
            return np.array(preds_aggregated)
        else:
            return preds
    
    def _save_model(self, model, filepath):
        # Create directory if it doesn't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the model
        torch.save({
            'model_state_dict': model.state_dict(),
        }, filepath)
    
    def _load_model(self, filepath):
        # Load the model
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)

def train_xresnet1d101(data_folder, output_folder, task='superdiagnostic', sampling_frequency=100, 
                       train_fold=8, val_fold=9, test_fold=10, epochs=50, batch_size=128):
    """
    Main function to train the xresnet1d101 model on ECG data.
    
    Parameters:
    - data_folder: Path to the data folder
    - output_folder: Path to save the model and results
    - task: Classification task type ('diagnostic', 'subdiagnostic', 'superdiagnostic', 'form', 'rhythm', 'all', or 'custom')
    - sampling_frequency: ECG sampling frequency in Hz (100 or 500)
    - train_fold: Fold index for training
    - val_fold: Fold index for validation
    - test_fold: Fold index for testing
    - epochs: Number of training epochs
    - batch_size: Batch size for training
    
    Returns:
    - trained model and evaluation results
    """
    # Import utility functions here to avoid circular imports
    from utils import load_dataset, compute_label_aggregations, select_data, preprocess_signals
    
    print(f"Loading data from {data_folder}")
    data, raw_labels = load_dataset(data_folder, sampling_frequency)
    
    print(f"Computing label aggregations for task: {task}")
    labels = compute_label_aggregations(raw_labels, os.path.join(data_folder, 'scp_statements.csv'), task)
    
    print("Selecting and preprocessing data")
    os.makedirs(output_folder, exist_ok=True)
    data_output_folder = os.path.join(output_folder, 'data')
    os.makedirs(data_output_folder, exist_ok=True)
    
    data_selected, labels_selected, Y, _ = select_data(data, labels, task, min_samples=0, outputfolder=data_output_folder)
    input_shape = data_selected[0].shape
    
    # Split data into train, val, test sets
    try:
        # Try to use strat_fold column if it exists (PTB-XL standard)
        X_test = data_selected[labels_selected.strat_fold == test_fold]
        y_test = Y[labels_selected.strat_fold == test_fold]
        X_val = data_selected[labels_selected.strat_fold == val_fold]
        y_val = Y[labels_selected.strat_fold == val_fold]
        X_train = data_selected[labels_selected.strat_fold <= train_fold]
        y_train = Y[labels_selected.strat_fold <= train_fold]
    except:
        # If strat_fold doesn't exist, split based on percentage
        print("No stratified folds found, splitting data by percentage")
        total_samples = len(data_selected)
        train_end = int(total_samples * 0.8)
        val_end = int(total_samples * 0.9)
        
        X_train = data_selected[:train_end]
        y_train = Y[:train_end]
        X_val = data_selected[train_end:val_end]
        y_val = Y[train_end:val_end]
        X_test = data_selected[val_end:]
        y_test = Y[val_end:]
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Number of classes: {Y.shape[1]}")
    
    # Preprocess signals
    X_train, X_val, X_test = preprocess_signals(X_train, X_val, X_test, data_output_folder)
    
    # Save labels for later evaluation
    np.save(os.path.join(data_output_folder, 'y_train.npy'), y_train)
    np.save(os.path.join(data_output_folder, 'y_val.npy'), y_val)
    np.save(os.path.join(data_output_folder, 'y_test.npy'), y_test)
    
    # Initialize and train the model
    model_name = 'xresnet1d101'
    model = FastaiModel(
        name=model_name,
        n_classes=Y.shape[1],
        freq=sampling_frequency,
        outputfolder=output_folder,
        input_shape=input_shape,
        epochs=epochs,
        bs=batch_size,
        early_stopping="valid_loss"
    )
    
    print("Starting model training")
    model.fit(X_train, y_train, X_val, y_val)
    
    # Evaluate on test set
    print("Evaluating model on test set")
    y_pred = model.predict(X_test)
    test_results = evaluate_experiment(y_test, y_pred)
    
    print("Test results:")
    for metric, value in test_results.items():
        print(f"{metric}: {value:.4f}")
    
    # Save test results
    test_results_df = pd.DataFrame(test_results, index=[0])
    os.makedirs(os.path.join(output_folder, 'results'), exist_ok=True)
    test_results_df.to_csv(os.path.join(output_folder, 'results', 'test_results.csv'))
    
    return model, test_results