import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

# Import XResNet model
from metrics.classification.model import xresnet1d101
from matplotlib.legend_handler import HandlerTuple

# Helper function for scientific notation
def format_sci(val):
    """Format values < 1e-4 in scientific notation"""
    if abs(val) < 1e-4 and val != 0:
        return f"{val:.2e}"
    else:
        return f"{val:.5f}"

# Custom formatter for matplotlib y-axis values
class ScientificFormatter(plt.ScalarFormatter):
    def __call__(self, x, pos=None):
        if abs(x) < 1e-4 and x != 0:
            return f"{x:.2e}"
        else:
            return f"{x:.5g}"  # Use general format for other values

class TXTY_Evaluator:
    def __init__(self, 
                 input_shape,
                 num_classes, 
                 model_params=None, 
                 device=None,
                 output_dir='./txty_results'):
        """
        Initialize TXTY Evaluator for training on X and testing on Y
        
        Args:
            input_shape: Shape of input data (channels, length)
            num_classes: Number of classes
            model_params: Dictionary of model parameters
            device: Torch device
            output_dir: Directory to save results
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_shape = input_shape
        self.num_classes = num_classes
        # Use paper hyperparameters
        self.model_params = model_params if model_params else {
            'ps_head': 0.5,            # Dropout rate from paper
            'lin_ftrs_head': [128],    # Hidden layer size from paper
            'kernel_size': 5,          # Kernel size from paper
        }
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.model = self._init_model()
        
    def _init_model(self):
        """Initialize XResNet model"""
        model = xresnet1d101(
            num_classes=self.num_classes,
            input_channels=self.input_shape[0],
            ps_head=self.model_params['ps_head'],
            lin_ftrs_head=self.model_params['lin_ftrs_head'],
            kernel_size=self.model_params['kernel_size']
        )
        model = model.to(self.device)
        return model
    
    def train(self, train_data, train_labels, val_data=None, val_labels=None, 
              batch_size=128, lr=1e-2, weight_decay=1e-2, epochs=50, save_best_path=None):
        """
        Train the model on the given training data
        
        Args:
            train_data: Training data tensor (B, C, L)
            train_labels: Training labels tensor (B, num_classes)
            val_data: Validation data tensor (B, C, L)
            val_labels: Validation labels tensor (B, num_classes)
            batch_size: Batch size for training (paper: 128)
            lr: Learning rate (paper: 1e-2)
            weight_decay: Weight decay (paper: 1e-2)
            epochs: Number of training epochs (paper: 50)
            save_best_path: Path to save the best model
            
        Returns:
            Dictionary of training history
            Path to the saved best model
        """
        # Reset model weights
        self.model = self._init_model()
        
        # Create dataloaders
        train_dataset = TensorDataset(train_data, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if val_data is not None and val_labels is not None:
            val_dataset = TensorDataset(val_data, val_labels)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        else:
            val_loader = None
        
        # Define loss function and optimizer (using AdamW as in paper)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Training loop
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_auroc': [],
            'val_auroc': []
        }
        
        best_val_auroc = 0
        best_model = None
        best_model_path = None
        
        for epoch in tqdm(range(epochs), desc="Epochs [Train]"):
            # Training phase
            self.model.train()
            train_loss = 0
            train_preds = []
            train_targets = []
            
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                # Store predictions and targets for AUROC calculation
                train_preds.append(torch.sigmoid(output).detach().cpu().numpy())
                train_targets.append(target.detach().cpu().numpy())
            
            # Calculate epoch metrics
            train_loss /= len(train_loader)
            train_preds = np.vstack(train_preds)
            train_targets = np.vstack(train_targets)
            train_auroc = self._calculate_macro_auroc(train_targets, train_preds)
            
            history['train_loss'].append(train_loss)
            history['train_auroc'].append(train_auroc)
            
            # Validation phase
            if val_loader:
                val_loss, val_auroc, val_preds = self.evaluate(val_data, val_labels, batch_size, use_tqdm=False)
                history['val_loss'].append(val_loss)
                history['val_auroc'].append(val_auroc)
                
                # Save best model
                if val_auroc > best_val_auroc:
                    best_val_auroc = val_auroc
                    best_model = self.model.state_dict().copy()
                
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train AUROC: {train_auroc:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val AUROC: {val_auroc:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train AUROC: {train_auroc:.4f}")
        
        # Load best model if validation was used
        if best_model is not None:
            self.model.load_state_dict(best_model)
            # Save best model to disk if requested
            if save_best_path is not None:
                torch.save(best_model, save_best_path)
                best_model_path = save_best_path
            
        # Save visualization of training progress
        self.visualize_training_progress(history)
            
        return history
    
    def visualize_training_progress(self, history):
        """
        Visualize and save training progress with thesis-quality styling

        Args:
            history: Dictionary containing training history
        """
        # Set up better styling for publication-quality figures
        plt.rcParams.update({
            'font.family': 'DejaVu Serif',
            'mathtext.fontset': 'cm',
            'font.size': 14,
            'axes.labelsize': 16,
            'xtick.labelsize': 14,
            'ytick.labelsize': 13,
            'legend.fontsize': 15,
            'figure.figsize': (12, 5.8),
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.05,
        })

        # Define colorblind-friendly colors
        colors = ['#000000', '#1F77B4']

        fig = plt.figure(figsize=(12, 5.8))
        subplot_height = 0.60
        subplot_bottom = 0.22

        ax1 = fig.add_axes([0.08, subplot_bottom, 0.42, subplot_height])
        ax1.yaxis.set_major_formatter(ScientificFormatter())
        ax2 = fig.add_axes([0.58, subplot_bottom, 0.42, subplot_height])

        epochs = range(1, len(history['train_loss']) + 1)
        l1, = ax1.plot(epochs, history['train_loss'], marker='o', color=colors[0],
                       linewidth=2, markersize=4, markeredgecolor='white', markeredgewidth=0.5)
        l2 = l4 = None
        if history['val_loss']:
            l2, = ax1.plot(epochs, history['val_loss'], marker='o', color=colors[1],
                           linewidth=2, markersize=4, markeredgecolor='white', markeredgewidth=0.5)
        if history['val_auroc']:
            l4, = ax2.plot(epochs, history['val_auroc'], marker='o', color=colors[1],
                           linewidth=2, markersize=4, markeredgecolor='white', markeredgewidth=0.5)
        l3, = ax2.plot(epochs, history['train_auroc'], marker='o', color=colors[0],
                       linewidth=2, markersize=4, markeredgecolor='white', markeredgewidth=0.5)

        min_loss_epoch = np.argmin(history['val_loss']) + 1 if history['val_loss'] else np.argmin(history['train_loss']) + 1
        band_width = 0.18
        band = ax1.axvspan(min_loss_epoch - band_width, min_loss_epoch + band_width, color='crimson', alpha=0.12, zorder=1)
        ax2.axvspan(min_loss_epoch - band_width, min_loss_epoch + band_width, color='crimson', alpha=0.12, zorder=1)

        for ax in [ax1, ax2]:
            ax.set_xlabel('Epoch', fontsize=16)
            ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
            for i in range(len(history['train_loss'])):
                ax.axvline(i+1, color='gray', linestyle='-', linewidth=0.5, alpha=0.2, zorder=0)
        ax1.set_ylabel('Loss', fontsize=16)
        ax2.set_ylabel('Macro-AUROC', fontsize=16)

        if history['train_auroc'] and min(history['train_auroc']) >= 0:
            ax2.set_ylim(bottom=0)
            if history['val_auroc'] and max(history['val_auroc']) <= 1:
                ax2.set_ylim(top=1.05)

        handles = [ (l1, l3) ]
        labels = ['Training']
        if l2 and l4:
            handles.append((l2, l4))
            labels.append('Validation')
        elif l2:
            handles.append(l2)
            labels.append('Validation')
        elif l4:
            handles.append(l4)
            labels.append('Validation')
        handles.append(band)
        labels.append(f'Lowest validation loss = {format_sci(min(history["val_loss"]))}' if history['val_loss'] else f'Lowest training loss = {format_sci(min(history["train_loss"]))}')

        legend_y = subplot_bottom - 0.20
        fig.legend(
            handles, labels,
            handler_map={tuple: HandlerTuple(ndivide=None)},
            loc='lower center',
            bbox_to_anchor=(0.5, legend_y),
            ncol=3, frameon=False, fontsize=15
        )

        plt.tight_layout()
        save_path = self.output_dir / "training_progress.png"
        plt.savefig(save_path, dpi=300, transparent=True)
        plt.savefig(self.output_dir / "training_progress.pdf", transparent=True)
        print(f"Training progress visualization saved to {save_path}")
        plt.close(fig)
    
    def evaluate(self, data, labels, batch_size=128, use_tqdm=True):
        """
        Evaluate the model on the given data
        
        Args:
            data: Data tensor (B, C, L)
            labels: Labels tensor (B, num_classes)
            batch_size: Batch size for evaluation
            
        Returns:
            loss: Average loss
            auroc: Macro AUROC score
            predictions: Model predictions
        """
        self.model.eval()
        
        dataset = TensorDataset(data, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        
        criterion = nn.BCEWithLogitsLoss()
        
        loss = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            data_iter = tqdm(dataloader, desc="Evaluation") if use_tqdm else dataloader
            for data, target in data_iter:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss += criterion(output, target).item()
                
                predictions.append(torch.sigmoid(output).cpu().numpy())
                targets.append(target.cpu().numpy())
        
        loss /= len(dataloader)
        predictions = np.vstack(predictions)
        targets = np.vstack(targets)
        auroc = self._calculate_macro_auroc(targets, predictions)
        
        return loss, auroc, predictions
    
    def _calculate_macro_auroc(self, y_true, y_pred):
        """Calculate macro AUROC score"""
        try:
            return roc_auc_score(y_true, y_pred, average='macro')
        except ValueError:
            # Handle edge cases where some classes might not have positive samples
            aucs = []
            for i in range(y_true.shape[1]):
                if (y_true[:, i] == 0).all() or (y_true[:, i] == 1).all():
                    # Skip classes with only one label
                    continue
                aucs.append(roc_auc_score(y_true[:, i], y_pred[:, i]))
            return np.mean(aucs) if aucs else 0.5


# Individual metric functions
def run_TRTR(train_data, train_labels, val_data, val_labels, **kwargs):
    """
    Run TRTR (Train on Real, Test on Real) evaluation
    
    Args:
        train_data: Real training data
        train_labels: Real training labels
        val_data: Real validation data
        val_labels: Real validation labels
        **kwargs: Additional parameters for training
        
    Returns:
        TRTR AUROC score at last epoch,
        AUROC at epoch with lowest validation loss,
        training history
    """
    device = kwargs.get('device', None)
    if device is None:
        device = train_data.device if hasattr(train_data, 'device') else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    input_shape = train_data.shape[1:]
    num_classes = train_labels.shape[1]
    
    # Get parameters with paper's defaults
    batch_size = kwargs.get('batch_size', 128)
    lr = kwargs.get('lr', 1e-2)
    weight_decay = kwargs.get('weight_decay', 1e-2)
    epochs = kwargs.get('epochs', 50)
    output_dir = kwargs.get('output_dir', './txty_results/TRTR')
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    evaluator = TXTY_Evaluator(input_shape, num_classes, device=device, output_dir=output_dir)
    best_model_path = os.path.join(output_dir, "best_model.pth")
    # If a best model exists, load it
    if os.path.exists(best_model_path):
        state_dict = torch.load(best_model_path, map_location=device)
        evaluator.model.load_state_dict(state_dict)
        print(f"Loaded TRTR model weights from {best_model_path}")
        # Only evaluate, do not train
        loss, auroc_last, _ = evaluator.evaluate(val_data, val_labels, batch_size)
        # No training history, so fill with empty or None
        history = {'train_loss': [], 'val_loss': [], 'train_auroc': [], 'val_auroc': []}
        min_loss_auroc = auroc_last
        return auroc_last, min_loss_auroc, history

    # Repeat training until all losses are <= 1
    max_loss = float('inf')
    attempt = 0
    while max_loss > 1.0:
        attempt += 1
        print(f"\nTRTR training attempt {attempt}...")
        history = evaluator.train(
            train_data, train_labels,
            val_data, val_labels,
            batch_size=batch_size, lr=lr, weight_decay=weight_decay, epochs=epochs,
            save_best_path=best_model_path
        )
        # Check all losses (train and val) for all epochs
        all_losses = []
        if history['train_loss']:
            all_losses.extend(history['train_loss'])
        if history['val_loss']:
            all_losses.extend(history['val_loss'])
        max_loss = max(all_losses) if all_losses else float('inf')
        if max_loss > 1.0:
            print(f"Max loss {max_loss:.4f} > 1.0, retraining...")
        else:
            print(f"Acceptable model found with max loss {max_loss:.4f}.")

    _, auroc_last, _ = evaluator.evaluate(val_data, val_labels, batch_size)
    if history['val_loss']:
        min_loss_epoch = np.argmin(history['val_loss'])
        min_loss_auroc = history['val_auroc'][min_loss_epoch]
    else:
        min_loss_epoch = np.argmin(history['train_loss'])
        min_loss_auroc = history['train_auroc'][min_loss_epoch]

    if auroc_last > min_loss_auroc:
        print(f"AUROC at last epoch ({auroc_last:.4f}) is greater than AUROC at epoch with lowest validation loss ({min_loss_auroc:.4f}).")
    return auroc_last, min_loss_auroc, history

def run_TSTR(model_train_data, model_train_labels, val_data, val_labels, **kwargs):
    """
    Run TSTR (Train on Synthetic, Test on Real) evaluation
    
    Args:
        model_train_data: Synthetic training data
        model_train_labels: Synthetic training labels
        val_data: Real validation data
        val_labels: Real validation labels
        **kwargs: Additional parameters for training
        
    Returns:
        TSTR AUROC score at last epoch,
        AUROC at epoch with lowest validation loss,
        training history
    """
    device = kwargs.get('device', None)
    if device is None:
        device = model_train_data.device if hasattr(model_train_data, 'device') else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    input_shape = model_train_data.shape[1:]
    num_classes = model_train_labels.shape[1]
    
    # Get parameters with paper's defaults
    batch_size = kwargs.get('batch_size', 128)
    lr = kwargs.get('lr', 1e-2)
    weight_decay = kwargs.get('weight_decay', 1e-2)
    epochs = kwargs.get('epochs', 50)
    output_dir = kwargs.get('output_dir', './txty_results/TSTR')
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize evaluator
    evaluator = TXTY_Evaluator(input_shape, num_classes, device=device, output_dir=output_dir)
    
    # Train and evaluate
    history = evaluator.train(
        model_train_data, model_train_labels, 
        val_data, val_labels,
        batch_size=batch_size, lr=lr, weight_decay=weight_decay, epochs=epochs
    )
    # Instead of evaluating again, just use the AUROC at the best val loss epoch
    if history['val_loss']:
        min_loss_epoch = np.argmin(history['val_loss'])
        min_loss_auroc = history['val_auroc'][min_loss_epoch]
        auroc_best = min_loss_auroc
    else:
        min_loss_epoch = np.argmin(history['train_loss'])
        min_loss_auroc = history['train_auroc'][min_loss_epoch]
        auroc_best = min_loss_auroc

    # Optionally print info
    # print(f"Best AUROC at min val loss epoch ({min_loss_epoch+1}): {auroc_best:.4f}")

    return auroc_best, min_loss_auroc, history

def run_TRTS(train_data, train_labels, model_val_data, model_val_labels, model_checkpoint_path=None, **kwargs):
    """
    Run TRTS (Train on Real, Test on Synthetic) evaluation
    
    Args:
        train_data: Real training data
        train_labels: Real training labels
        model_val_data: Synthetic validation data
        model_val_labels: Synthetic validation labels
        model_checkpoint_path: Path to the saved model checkpoint
        **kwargs: Additional parameters for training
        
    Returns:
        TRTS AUROC score at last epoch,
        AUROC at epoch with lowest validation loss,
        training history
    """
    device = kwargs.get('device', None)
    if device is None:
        device = train_data.device if hasattr(train_data, 'device') else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    input_shape = train_data.shape[1:]
    num_classes = train_labels.shape[1]
    
    # Get parameters with paper's defaults
    batch_size = kwargs.get('batch_size', 128)
    lr = kwargs.get('lr', 1e-2)
    weight_decay = kwargs.get('weight_decay', 1e-2)
    epochs = kwargs.get('epochs', 50)
    output_dir = kwargs.get('output_dir', './txty_results/TRTS')
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize evaluator
    evaluator = TXTY_Evaluator(input_shape, num_classes, device=device, output_dir=output_dir)
    # Optionally load model weights from checkpoint
    if model_checkpoint_path is not None and os.path.exists(model_checkpoint_path):
        state_dict = torch.load(model_checkpoint_path, map_location=device)
        evaluator.model.load_state_dict(state_dict)
        print(f"Loaded TRTR model weights from {model_checkpoint_path}")
        # Only evaluate, do not train
        loss, auroc_last, _ = evaluator.evaluate(model_val_data, model_val_labels, batch_size)
        # No training history, so fill with empty or None
        history = {'train_loss': [], 'val_loss': [], 'train_auroc': [], 'val_auroc': []}
        min_loss_auroc = auroc_last
        return auroc_last, min_loss_auroc, history

    history = evaluator.train(
        train_data, train_labels, 
        model_val_data, model_val_labels,
        batch_size=batch_size, lr=lr, weight_decay=weight_decay, epochs=epochs
    )
    _, auroc_last, _ = evaluator.evaluate(model_val_data, model_val_labels, batch_size)
    if history['val_loss']:
        min_loss_epoch = np.argmin(history['val_loss'])
        min_loss_auroc = history['val_auroc'][min_loss_epoch]
    else:
        min_loss_epoch = np.argmin(history['train_loss'])
        min_loss_auroc = history['train_auroc'][min_loss_epoch]
    if auroc_last > min_loss_auroc:
        print(f"AUROC at last epoch ({auroc_last:.4f}) is greater than AUROC at epoch with lowest validation loss ({min_loss_auroc:.4f}).")
    return auroc_last, min_loss_auroc, history

# Main evaluation function for integration
def evaluate(log_dict, real_train, real_val, fake_train, fake_val, **kwargs):
    """
    Evaluate with TXTY metrics
    
    Args:
        log_dict: Dictionary to store results
        real_train: Tuple of (train_data, train_labels, train_features)
        real_val: Tuple of (val_data, val_labels, val_features)
        fake_train: Tuple of (model_train_data, model_train_labels, model_train_features)
        fake_val: Tuple of (model_val_data, model_val_labels, model_val_features)
        **kwargs: Additional parameters for evaluation
        
    Returns:
        Updated log_dict with metrics
    """
    train_data, train_labels, _ = real_train
    val_data, val_labels, _ = real_val
    model_train_data, model_train_labels, _ = fake_train
    model_val_data, model_val_labels, _ = fake_val

    # Set up output directories
    base_output_dir = kwargs.get('output_dir', './txty_results')
    Path(base_output_dir).mkdir(parents=True, exist_ok=True)
    
    kwargs['output_dir'] = f"{base_output_dir}/TRTR"
    
    # Group for TSTR
    TSTR_metric_data = (model_train_data, model_train_labels, val_data, val_labels)
    # Group for TRTS
    TRTS_metric_data = (train_data, train_labels, model_val_data, model_val_labels)
    # Group for baseline
    classifier_baseline_data = (train_data, train_labels, val_data, val_labels)

    # Run the metrics
    print("\n=== Running TRTR (Train on Real, Test on Real) - Baseline Evaluation ===")
    TRTR_score, TRTR_minloss_auroc, TRTR_history, trtr_model_path = run_TRTR(*classifier_baseline_data, **kwargs)
    
    kwargs['output_dir'] = f"{base_output_dir}/TSTR"
    print("\n=== Running TSTR (Train on Synthetic, Test on Real) Evaluation ===")
    TSTR_score, TSTR_minloss_auroc, TSTR_history = run_TSTR(*TSTR_metric_data, **kwargs)
    
    kwargs['output_dir'] = f"{base_output_dir}/TRTS"
    print("\n=== Running TRTS (Train on Real, Test on Synthetic) Evaluation ===")
    TRTS_score, TRTS_minloss_auroc, TRTS_history = run_TRTS(*TRTS_metric_data, model_checkpoint_path=trtr_model_path, **kwargs)
    
    # Store results in log_dict
    log_dict["TRTR_auroc"] = TRTR_minloss_auroc
    log_dict["TSTR_auroc"] = TSTR_minloss_auroc
    log_dict["TRTS_auroc"] = TRTS_minloss_auroc
    
    # Calculate utility scores
    # log_dict["TSTR_utility"] = TSTR_score / TRTR_score if TRTR_score > 0 else 0
    # log_dict["TRTS_utility"] = TRTS_score / TRTR_score if TRTR_score > 0 else 0
    
    # Print summary
    print("\n=== TXTY Evaluation Summary ===")
    print(f"TRTR (baseline): {TRTR_score:.4f} (min-loss AUROC: {TRTR_minloss_auroc:.4f})")
    print(f"TSTR: {TSTR_score:.4f} (min-loss AUROC: {TSTR_minloss_auroc:.4f}, Utility: {log_dict['TSTR_utility']:.4f})")
    print(f"TRTS: {TRTS_score:.4f} (min-loss AUROC: {TRTS_minloss_auroc:.4f}, Utility: {log_dict['TRTS_utility']:.4f})")
    
    return log_dict