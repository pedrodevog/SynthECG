import os
import numpy as np
import pandas as pd
import pickle
import ast
import wfdb
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.metrics import roc_auc_score

def evaluate_experiment(y_true, y_pred, thresholds=None):
    """Evaluate the model predictions against true labels"""
    results = {}
    
    # label based metric
    results['macro_auc'] = roc_auc_score(y_true, y_pred, average='macro')
    
    # Calculate F-max metric
    precisions, recalls, thrs = precision_recall_curve(y_true, y_pred)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-10)
    results['Fmax'] = np.max(f1s)
    
    return results

def precision_recall_curve(y_true, y_pred):
    """Calculate precision-recall curve for multi-label classification"""
    # Initialize arrays
    precisions = []
    recalls = []
    thresholds = np.linspace(0, 1, 100)
    
    # For each threshold
    for thr in thresholds:
        y_pred_binary = (y_pred > thr).astype(int)
        
        # Calculate precision and recall
        precision = np.sum((y_pred_binary == 1) & (y_true == 1)) / (np.sum(y_pred_binary == 1) + 1e-10)
        recall = np.sum((y_pred_binary == 1) & (y_true == 1)) / (np.sum(y_true == 1) + 1e-10)
        
        precisions.append(precision)
        recalls.append(recall)
    
    return np.array(precisions), np.array(recalls), thresholds

def load_dataset(path, sampling_rate=100):
    """Load the PTB-XL dataset or a custom dataset"""
    try:
        # Try to load as PTB-XL dataset
        Y = pd.read_csv(path + '/ptbxl_database.csv', index_col='ecg_id')
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
        
        # Load raw signal data
        X = load_raw_data_ptbxl(Y, sampling_rate, path)
    except:
        # Handle as custom dataset
        print("Custom dataset detected, attempting to load data...")
        X = np.load(path + '/data.npy', allow_pickle=True)
        Y = pd.read_csv(path + '/labels.csv')
        
    return X, Y

def load_raw_data_ptbxl(df, sampling_rate, path):
    """Load raw PTB-XL dataset"""
    if sampling_rate == 100:
        if os.path.exists(path + '/raw100.npy'):
            data = np.load(path + '/raw100.npy', allow_pickle=True)
        else:
            data = [wfdb.rdsamp(path + '/' + f) for f in tqdm(df.filename_lr)]
            data = np.array([signal for signal, meta in data])
            np.save(path + '/raw100.npy', data)
    elif sampling_rate == 500:
        if os.path.exists(path + '/raw500.npy'):
            data = np.load(path + '/raw500.npy', allow_pickle=True)
        else:
            data = [wfdb.rdsamp(path + '/' + f) for f in tqdm(df.filename_hr)]
            data = np.array([signal for signal, meta in data])
            np.save(path + '/raw500.npy', data)
    return data

def compute_label_aggregations(df, annotation_file, ctype):
    """Compute aggregated labels based on the specified classification type"""
    df['scp_codes_len'] = df.scp_codes.apply(lambda x: len(x))
    
    # Load the annotation file with SCP statement mappings
    try:
        aggregation_df = pd.read_csv(annotation_file, index_col=0)
    except:
        # If annotation file doesn't exist, return original df (for custom datasets)
        print("Annotation file not found. Using original labels.")
        return df
    
    if ctype in ['diagnostic', 'subdiagnostic', 'superdiagnostic']:
        diag_agg_df = aggregation_df[aggregation_df.diagnostic == 1.0]
        
        # Define aggregation functions
        def aggregate_all_diagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in diag_agg_df.index:
                    tmp.append(key)
            return list(set(tmp))
        
        def aggregate_subdiagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in diag_agg_df.index:
                    c = diag_agg_df.loc[key].diagnostic_subclass
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))
        
        def aggregate_diagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in diag_agg_df.index:
                    c = diag_agg_df.loc[key].diagnostic_class
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))
        
        # Apply aggregation
        if ctype == 'diagnostic':
            df['diagnostic'] = df.scp_codes.apply(aggregate_all_diagnostic)
            df['diagnostic_len'] = df.diagnostic.apply(lambda x: len(x))
        elif ctype == 'subdiagnostic':
            df['subdiagnostic'] = df.scp_codes.apply(aggregate_subdiagnostic)
            df['subdiagnostic_len'] = df.subdiagnostic.apply(lambda x: len(x))
        elif ctype == 'superdiagnostic':
            df['superdiagnostic'] = df.scp_codes.apply(aggregate_diagnostic)
            df['superdiagnostic_len'] = df.superdiagnostic.apply(lambda x: len(x))
            
    elif ctype == 'form':
        form_agg_df = aggregation_df[aggregation_df.form == 1.0]
        
        def aggregate_form(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in form_agg_df.index:
                    tmp.append(key)
            return list(set(tmp))
            
        df['form'] = df.scp_codes.apply(aggregate_form)
        df['form_len'] = df.form.apply(lambda x: len(x))
        
    elif ctype == 'rhythm':
        rhythm_agg_df = aggregation_df[aggregation_df.rhythm == 1.0]
        
        def aggregate_rhythm(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in rhythm_agg_df.index:
                    tmp.append(key)
            return list(set(tmp))
            
        df['rhythm'] = df.scp_codes.apply(aggregate_rhythm)
        df['rhythm_len'] = df.rhythm.apply(lambda x: len(x))
        
    elif ctype == 'all':
        df['all_scp'] = df.scp_codes.apply(lambda x: list(set(x.keys())))
        
    return df

def select_data(X, Y, ctype, min_samples=0, outputfolder=None):
    """Select data based on the specified classification type and convert labels to one-hot encoding"""
    # Convert multilabel to multi-hot
    mlb = MultiLabelBinarizer()
    
    # Handle custom labels
    if ctype == 'custom':
        if 'label' in Y.columns:
            # Assume label is already in correct format
            y = np.array(Y['label'].tolist())
            return X, Y, y, None
            
    # Process standard PTB-XL classification types
    if ctype == 'diagnostic':
        X_selected = X[Y.diagnostic_len > 0]
        Y_selected = Y[Y.diagnostic_len > 0]
        mlb.fit(Y_selected.diagnostic.values)
        y = mlb.transform(Y_selected.diagnostic.values)
    elif ctype == 'subdiagnostic':
        counts = pd.Series(np.concatenate(Y.subdiagnostic.values)).value_counts()
        counts = counts[counts > min_samples]
        Y.subdiagnostic = Y.subdiagnostic.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        Y['subdiagnostic_len'] = Y.subdiagnostic.apply(lambda x: len(x))
        X_selected = X[Y.subdiagnostic_len > 0]
        Y_selected = Y[Y.subdiagnostic_len > 0]
        mlb.fit(Y_selected.subdiagnostic.values)
        y = mlb.transform(Y_selected.subdiagnostic.values)
    elif ctype == 'superdiagnostic':
        counts = pd.Series(np.concatenate(Y.superdiagnostic.values)).value_counts()
        counts = counts[counts > min_samples]
        Y.superdiagnostic = Y.superdiagnostic.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        Y['superdiagnostic_len'] = Y.superdiagnostic.apply(lambda x: len(x))
        X_selected = X[Y.superdiagnostic_len > 0]
        Y_selected = Y[Y.superdiagnostic_len > 0]
        mlb.fit(Y_selected.superdiagnostic.values)
        y = mlb.transform(Y_selected.superdiagnostic.values)
    elif ctype == 'form':
        counts = pd.Series(np.concatenate(Y.form.values)).value_counts()
        counts = counts[counts > min_samples]
        Y.form = Y.form.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        Y['form_len'] = Y.form.apply(lambda x: len(x))
        X_selected = X[Y.form_len > 0]
        Y_selected = Y[Y.form_len > 0]
        mlb.fit(Y_selected.form.values)
        y = mlb.transform(Y_selected.form.values)
    elif ctype == 'rhythm':
        counts = pd.Series(np.concatenate(Y.rhythm.values)).value_counts()
        counts = counts[counts > min_samples]
        Y.rhythm = Y.rhythm.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        Y['rhythm_len'] = Y.rhythm.apply(lambda x: len(x))
        X_selected = X[Y.rhythm_len > 0]
        Y_selected = Y[Y.rhythm_len > 0]
        mlb.fit(Y_selected.rhythm.values)
        y = mlb.transform(Y_selected.rhythm.values)
    elif ctype == 'all':
        counts = pd.Series(np.concatenate(Y.all_scp.values)).value_counts()
        counts = counts[counts > min_samples]
        Y.all_scp = Y.all_scp.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        Y['all_scp_len'] = Y.all_scp.apply(lambda x: len(x))
        X_selected = X[Y.all_scp_len > 0]
        Y_selected = Y[Y.all_scp_len > 0]
        mlb.fit(Y_selected.all_scp.values)
        y = mlb.transform(Y_selected.all_scp.values)
    else:
        X_selected = X
        Y_selected = Y
        y = Y  # Placeholder, should be replaced with actual label processing for custom data
    
    # Save LabelBinarizer if outputfolder is provided
    if outputfolder:
        os.makedirs(outputfolder, exist_ok=True)
        with open(os.path.join(outputfolder, 'mlb.pkl'), 'wb') as f:
            pickle.dump(mlb, f)
    
    return X_selected, Y_selected, y, mlb

def preprocess_signals(X_train, X_validation, X_test, outputfolder=None):
    """Standardize ECG signals"""
    # Standardize data such that mean 0 and variance 1
    ss = StandardScaler()
    ss.fit(np.vstack(X_train).flatten()[:,np.newaxis].astype(float))
    
    # Save Standardizer data if outputfolder is provided
    if outputfolder:
        os.makedirs(outputfolder, exist_ok=True)
        with open(os.path.join(outputfolder, 'standard_scaler.pkl'), 'wb') as f:
            pickle.dump(ss, f)
    
    return apply_standardizer(X_train, ss), apply_standardizer(X_validation, ss), apply_standardizer(X_test, ss)

def apply_standardizer(X, ss):
    """Apply standardization to signals"""
    X_tmp = []
    for x in X:
        x_shape = x.shape
        X_tmp.append(ss.transform(x.flatten()[:,np.newaxis]).reshape(x_shape))
    X_tmp = np.array(X_tmp)
    return X_tmp