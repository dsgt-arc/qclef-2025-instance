from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np

def k_fold(X, y, nr_folds=5, train_percentage=0.9, random_state=123, printstats=True): #in the paper they have 9:1 ratio for train to validation for each fold and k = 5 (and hence test = 20%)
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
    results = []
    kf = StratifiedKFold(n_splits=nr_folds, shuffle=True, random_state=random_state)
    
    if printstats:
        print(f"Full data:")
        print(f"  Size X: {X.shape}; Size Y: {y.shape}")
        print(f"  Class distribution in target variable: {np.bincount(y) / len(y)}")


    for i, (train_val_idx, test_idx) in enumerate(kf.split(X, y)):
        X_train_val, X_test = X[train_val_idx], X[test_idx]
        y_train_val, y_test = y[train_val_idx], y[test_idx]
    
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, train_size=train_percentage, stratify=y_train_val, random_state=random_state)
        
        if printstats:
            print(f"Fold {i+1}:")
            print(f"  Size X_train: {X_train.shape}; Y_train: {y_train.shape}")
            print(f"  Size X_val: {X_val.shape}; Y_val: {y_val.shape}")
            print(f"  Size X_test: {X_test.shape}; Y_test: {y_test.shape}")
            print(f"  Train class distribution in target variable: {np.bincount(y_train) / len(y_train)}")
            print(f"  Val class distribution in target variable: {np.bincount(y_val) / len(y_val)}")
            print(f"  Test class distribution in target variable: {np.bincount(y_test) / len(y_test)}\n")
        
        results.append((X_train, y_train, X_val, y_val, X_test, y_test))
    
    return results