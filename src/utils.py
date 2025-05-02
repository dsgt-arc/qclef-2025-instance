from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np

def k_fold(X_raw_text, y_raw_text, X_embeddings, y_embeddings, nr_folds=5, train_percentage=0.9, random_state=123, printstats=True): #in the paper they have 9:1 ratio for train to validation for each fold and k = 5 (and hence test = 20%)
    """
    Peforms k-fold split like described in Figure 3 from the Pasin 2024 paper. For each fold:
     - Test sets are used for true OOS evaluation of the predictive model only
     - validation sets are reserved for validating the predictive model only.
     - Instance selection is ran on the train set of each fold 

    Parameters:
    X (ndarray): Feature matrix.
    y (ndarray): Target array.
    k (int): Number of folds. This will dictate the size of the test set.
    train_percentage (float): train percentage of the data per fold, after setting aside the data of the test set
    
    Returns:
    List of tuples: Each tuple contains (X_train, y_train, X_val, y_val, X_test, y_test) for each fold.
    """
    raw_text_splits = []
    embeddings_splits = []
    kf = StratifiedKFold(n_splits=nr_folds, shuffle=True, random_state=random_state)
    
    if printstats:
        print(f"Full data:")
        print(f"  Size X: {X_raw_text.shape}; Size Y: {y_raw_text.shape}")
        print(f"  Class distribution in target variable: {np.bincount(y_raw_text) / len(y_raw_text)}")


    for i, (train_val_idx, test_idx) in enumerate(kf.split(X_raw_text, y_raw_text)):
        X_train_val_raw_text, X_test_raw_text = X_raw_text[train_val_idx], X_raw_text[test_idx]
        y_train_val_raw_text, y_test_raw_text = y_raw_text[train_val_idx], y_raw_text[test_idx]
        
        indices = np.arange(len(X_train_val_raw_text))
          
        X_train_val_embeddings, X_test_embeddings = X_embeddings[train_val_idx], X_embeddings[test_idx]
        y_train_val_embeddings, y_test_embeddings = y_embeddings[train_val_idx], y_embeddings[test_idx]
     
        X_train_raw_text, X_val_raw_text, y_train_raw_text, y_val_raw_text, train_indices, val_indices = train_test_split(
            X_train_val_raw_text, y_train_val_raw_text, indices, train_size=train_percentage, stratify=y_train_val_raw_text, random_state=random_state)
        
        X_train_embeddings = X_train_val_embeddings[train_indices]
        X_val_embeddings = X_train_val_embeddings[val_indices]
        y_train_embeddings = y_train_val_embeddings[train_indices]
        y_val_embeddings = y_train_val_embeddings[val_indices]
    
        if printstats:
            print(f"Fold {i+1}:")
            print(f"  Size X_train: {X_train_raw_text.shape}; Y_train: {y_train_raw_text.shape}")
            print(f"  Size X_val: {X_val_raw_text.shape}; Y_val: {y_val_raw_text.shape}")
            print(f"  Size X_test: {X_test_raw_text.shape}; Y_test: {y_test_raw_text.shape}")
            print(f"  Train class distribution in target variable: {np.bincount(y_train_raw_text) / len(y_train_raw_text)}")
            print(f"  Val class distribution in target variable: {np.bincount(y_val_raw_text) / len(y_val_raw_text)}")
            print(f"  Test class distribution in target variable: {np.bincount(y_test_raw_text) / len(y_test_raw_text)}\n")
        
        raw_text_splits.append((X_train_raw_text, y_train_raw_text, X_val_raw_text, y_val_raw_text, X_test_raw_text, y_test_raw_text))
        embeddings_splits.append((X_train_embeddings, y_train_embeddings, X_val_embeddings, y_val_embeddings, X_test_embeddings, y_test_embeddings))
    return raw_text_splits, embeddings_splits