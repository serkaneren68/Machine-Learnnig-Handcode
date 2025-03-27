import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load the breast cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target
target_names = data.target_names

# Function to find optimal threshold based on ROC curve (Fawcett approach)
def find_optimal_threshold(fpr, tpr, thresholds):
    """
    Find the optimal threshold that maximizes the distance to the diagonal line
    as described by Fawcett.
    """
    # Calculate the distance from each point to the diagonal line
    distances = tpr - fpr
    optimal_idx = np.argmax(distances)
    return thresholds[optimal_idx]

# Implement k-fold cross-validation with ROC curves
def svm_cross_validation(X, y, n_splits=5):
    # Start timing
    start_time = time.time()
    
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Initialize arrays to store results
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    confusion_matrices = []
    thresholds_list = []
    fold_times = []
    
    # Create figure for ROC curves
    plt.figure(figsize=(10, 8))
    
    # Create pipeline with scaling (important for SVM)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', LinearSVC(dual="auto", max_iter=10000, random_state=42))
    ])
    
    # Perform cross-validation
    for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train and time the model
        fold_start_time = time.time()
        pipeline.fit(X_train, y_train)
        
        # Get decision function scores (distance from hyperplane)
        y_score = pipeline.decision_function(X_test)
        fold_end_time = time.time()
        fold_time = fold_end_time - fold_start_time
        fold_times.append(fold_time)
        
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_score)
        
        # Find optimal threshold
        optimal_threshold = find_optimal_threshold(fpr, tpr, thresholds)
        thresholds_list.append(optimal_threshold)
        
        # Calculate predictions using the optimal threshold
        y_pred = (y_score >= optimal_threshold).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        confusion_matrices.append(cm)
        
        # Interpolate TPR at mean FPR points for averaging
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        
        # Calculate AUC
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        
        # Plot ROC curve for this fold
        plt.plot(fpr, tpr, lw=1, alpha=0.6,
                 label=f'ROC fold {i+1} (AUC = {roc_auc:.3f}, thresh = {optimal_threshold:.3f})')
        
        print(f"Fold {i+1}: AUC = {roc_auc:.3f}, Optimal threshold = {optimal_threshold:.3f}")
        print(f"Fold {i+1} training time: {fold_time:.3f} seconds")
        print(f"Confusion Matrix (threshold = {optimal_threshold:.3f}):")
        print(cm)
        print("-" * 50)
    
    # Plot the diagonal reference line
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray',
             label='Chance', alpha=.8)
    
    # Calculate and plot mean ROC curve
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)
    
    # Plot standard deviation area
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')
    
    # Calculate average optimal threshold
    avg_threshold = np.mean(thresholds_list)
    
    # Format plot
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves for Linear SVM (Mean AUC = {mean_auc:.3f})\nAvg Optimal Threshold = {avg_threshold:.3f}')
    plt.legend(loc="lower right")
    
    # Compute average confusion matrix
    avg_cm = np.mean(confusion_matrices, axis=0)
    
    # Plot average confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(avg_cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Average Confusion Matrix (Threshold = {avg_threshold:.3f})')
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    
    # Add text annotations to the confusion matrix
    thresh = avg_cm.max() / 2.
    for i in range(avg_cm.shape[0]):
        for j in range(avg_cm.shape[1]):
            plt.text(j, i, f"{avg_cm[i, j]:.1f}",
                    horizontalalignment="center",
                    color="white" if avg_cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # End timing
    end_time = time.time()
    total_time = end_time - start_time
    
    # Print performance metrics
    print("\nPerformance Summary:")
    print(f"Average AUC: {mean_auc:.3f} ± {std_auc:.3f}")
    print(f"Average Optimal Threshold: {avg_threshold:.3f}")
    print(f"Average Training Time per Fold: {np.mean(fold_times):.3f} seconds")
    print(f"Total Runtime: {total_time:.3f} seconds")
    
    # Show plots
    plt.show()
    
    return mean_auc, avg_threshold, avg_cm, np.mean(fold_times), total_time

# Run the cross-validation
print("Running 5-fold cross-validation on Linear SVM for Breast Cancer Wisconsin dataset...")
mean_auc, opt_threshold, avg_cm, avg_time, total_time = svm_cross_validation(X, y, n_splits=5)