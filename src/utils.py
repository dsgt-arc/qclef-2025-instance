from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np

def k_fold(X, y, k=5, train_val_ratio=[0.7, 0.1], printstats=True): #in the paper they have 7:1 ratio and k = 5
    """
    
    Parameters:
    X (ndarray): Feature matrix.
    y (ndarray): Target array.
    k (int): Number of folds.
    val_train_ratio (list): Relative sizes of training and validation sets, remainder is for test
    
    Returns:
    List of tuples: Each tuple contains (X_train, y_train, X_val, y_val, X_test, y_test) for each fold.
    """
    results = []
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=123)
    
    if printstats:
        print(f"Full data \n Class distribution: {np.bincount(y) / len(y)}")


    for i, (train_val_idx, test_idx) in enumerate(kf.split(X, y)):
        X_train_val, X_test = X[train_val_idx], X[test_idx]
        y_train_val, y_test = y[train_val_idx], y[test_idx]
        
        val_size = train_val_ratio[1] / sum(train_val_ratio)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, train_size=1 - val_size, stratify=y_train_val, random_state=42)
        
        if printstats:
            print(f"Fold {i+1}:")
            print(f"  Train class distribution: {np.bincount(y_train) / len(y_train)}")
            print(f"  Val class distribution: {np.bincount(y_val) / len(y_val)}")
            print(f"  Test class distribution: {np.bincount(y_test) / len(y_test)}\n")
        
        results.append((X_train, y_train, X_val, y_val, X_test, y_test))
    
    return results