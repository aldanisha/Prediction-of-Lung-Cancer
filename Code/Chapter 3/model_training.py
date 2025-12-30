import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report, 
                            confusion_matrix, roc_curve, auc, f1_score,
                            precision_score, recall_score)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("STARTING LUNG CANCER PREDICTION MODEL TRAINING")
print("="*60)

print("\n1. Loading and preparing data...")
df = pd.read_csv('lung_cancer_preprocessed_data.csv')
print(f"   Dataset shape: {df.shape}")
print(f"   Columns: {list(df.columns)}")

# Verify target columns exist
if 'LUNG_CANCER_YES' not in df.columns or 'LUNG_CANCER_NO' not in df.columns:
    print("   ❌ ERROR: Target columns not found!")
    print("   Available columns:", list(df.columns))
    exit()

# Prepare features and target
y = df['LUNG_CANCER_YES']  # Binary: 1=Cancer, 0=No Cancer
X = df.drop(['LUNG_CANCER_YES', 'LUNG_CANCER_NO'], axis=1)

print(f"   Features (X): {X.shape}")
print(f"   Target (y): {y.shape}")
print(f"   Class distribution: Cancer={y.sum()} ({y.mean():.1%}), No Cancer={(1-y).sum()} ({1-y.mean():.1%})")

print("\n2. Splitting data into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Training set: {X_train.shape[0]} samples")
print(f"   Testing set: {X_test.shape[0]} samples")

print("\n3. Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("   ✅ Features scaled successfully")

print("\n4. Handling class imbalance with SMOTE...")
print(f"   Before SMOTE - Training set class distribution:")
print(f"      Class 0 (No Cancer): {sum(y_train==0)} samples")
print(f"      Class 1 (Cancer): {sum(y_train==1)} samples")

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

print(f"   After SMOTE - Training set class distribution:")
print(f"      Class 0 (No Cancer): {sum(y_train_balanced==0)} samples")
print(f"      Class 1 (Cancer): {sum(y_train_balanced==1)} samples")

print("\n" + "="*60)
print("TRAINING MACHINE LEARNING MODELS")
print("="*60)

# UPDATED MODELS DICTIONARY WITH L2 REGULARIZATION
models = {
    'Logistic Regression (L2)': LogisticRegression(
        penalty='l2', 
        C=1.0, 
        max_iter=1000, 
        random_state=42
    ),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42)
}

results = {}
for name, model in models.items():
    print(f"\n   Training {name}...")
    # Train on BALANCED data
    model.fit(X_train_balanced, y_train_balanced)
    
    # Predict on test set
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    print(f"      Accuracy:  {accuracy:.4f}")
    print(f"      Precision: {precision:.4f}")
    print(f"      Recall:    {recall:.4f}")
    print(f"      F1-Score:  {f1:.4f}")

# ============================================================================
# NEW SECTION 1: OVER-FITTING CHECK (Required for 3.3.1)
# ============================================================================
print("\n" + "="*60)
print("MODEL EVALUATION: OVERFITTING CHECK")
print("="*60)

# Calculate train vs test performance for each model
print("\nChecking for overfitting (Train vs Test Accuracy):")
print("-" * 60)
overfitting_results = []
for name in models.keys():
    model = results[name]['model']
    
    # Predict on training data (balanced)
    y_train_pred = model.predict(X_train_balanced)
    train_acc = accuracy_score(y_train_balanced, y_train_pred)
    
    # Test accuracy from earlier
    test_acc = results[name]['accuracy']
    
    # Difference
    diff = train_acc - test_acc
    
    # Assessment
    if diff > 0.05:
        assessment = "Possible overfitting"
    elif diff < -0.02:
        assessment = "Possible underfitting"
    else:
        assessment = "Good generalization"
    
    overfitting_results.append({
        'Model': name,
        'Train Accuracy': train_acc,
        'Test Accuracy': test_acc,
        'Difference': diff,
        'Assessment': assessment
    })
    
    print(f"{name:<25} Train: {train_acc:.4f} | Test: {test_acc:.4f} | Diff: {diff:+.4f} -> {assessment}")

# Create overfitting comparison table
print("\n" + "-"*60)
print("Overfitting Analysis Summary:")
print("-"*60)
print(f"{'Model':<25} {'Train Acc':<12} {'Test Acc':<12} {'Diff':<12} {'Assessment':<20}")
print("-"*60)
for result in overfitting_results:
    print(f"{result['Model']:<25} {result['Train Accuracy']:<12.4f} {result['Test Accuracy']:<12.4f} "
          f"{result['Difference']:<+12.4f} {result['Assessment']:<20}")

# ============================================================================
# NEW SECTION 2: GRID SEARCH (Required for 3.3.3)
# ============================================================================
print("\n" + "="*60)
print("GRID SEARCH HYPERPARAMETER TUNING")
print("="*60)

# Perform Grid Search for SVM
print("\n1. Performing Grid Search for SVM...")
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'linear']
}

svm_for_grid = SVC(probability=True, random_state=42)
grid_search = GridSearchCV(
    svm_for_grid, 
    param_grid, 
    refit=True, 
    cv=5, 
    scoring='f1',
    verbose=0,
    n_jobs=-1
)

grid_search.fit(X_train_balanced, y_train_balanced)

print(f"   Best parameters found: {grid_search.best_params_}")
print(f"   Best cross-validation F1 score: {grid_search.best_score_:.4f}")

# Train final model with best parameters
best_svm = grid_search.best_estimator_
y_pred_best = best_svm.predict(X_test_scaled)
best_accuracy = accuracy_score(y_test, y_pred_best)
best_precision = precision_score(y_test, y_pred_best)
best_recall = recall_score(y_test, y_pred_best)
best_f1 = f1_score(y_test, y_pred_best)

print(f"   Test accuracy with tuned parameters: {best_accuracy:.4f}")
print(f"   Test precision with tuned parameters: {best_precision:.4f}")
print(f"   Test recall with tuned parameters: {best_recall:.4f}")
print(f"   Test F1-score with tuned parameters: {best_f1:.4f}")

# Compare with default SVM
if 'SVM' in results:
    default_svm_acc = results['SVM']['accuracy']
    default_svm_f1 = results['SVM']['f1']
    improvement_acc = best_accuracy - default_svm_acc
    improvement_f1 = best_f1 - default_svm_f1
    print(f"\n   Comparison with default SVM:")
    print(f"   Accuracy improvement: {improvement_acc:+.4f}")
    print(f"   F1-score improvement: {improvement_f1:+.4f}")

# Store best SVM results
results['SVM (Grid Search)'] = {
    'model': best_svm,
    'y_pred': y_pred_best,
    'accuracy': best_accuracy,
    'precision': best_precision,
    'recall': best_recall,
    'f1': best_f1
}

# ============================================================================
# NEW SECTION 3: THRESHOLD ADJUSTMENT (Required for 3.3.4)
# ============================================================================
print("\n" + "="*60)
print("MODEL REFINEMENT: THRESHOLD ADJUSTMENT")
print("="*60)

# Use the best SVM model
print("\n1. Testing different classification thresholds for SVM:")

# Get probability predictions instead of binary
y_prob = best_svm.predict_proba(X_test_scaled)[:, 1]

# Try different thresholds
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
threshold_results = []

print("-" * 70)
print(f"{'Threshold':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
print("-" * 70)

for thresh in thresholds:
    y_pred_thresh = (y_prob >= thresh).astype(int)
    acc = accuracy_score(y_test, y_pred_thresh)
    prec = precision_score(y_test, y_pred_thresh)
    rec = recall_score(y_test, y_pred_thresh)
    f1 = f1_score(y_test, y_pred_thresh)
    
    threshold_results.append({
        'Threshold': thresh,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1
    })
    
    print(f"{thresh:<12.1f} {acc:<12.4f} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f}")

# Find best threshold based on F1-score
best_threshold_result = max(threshold_results, key=lambda x: x['F1-Score'])
print(f"\n2. Best threshold: {best_threshold_result['Threshold']} (F1-Score: {best_threshold_result['F1-Score']:.4f})")

# ============================================================================
# NEW VISUALIZATION: THRESHOLD ANALYSIS PLOT
# ============================================================================
print("\n   5. Creating threshold analysis visualization...")

plt.figure(figsize=(12, 8))
thresholds = [r['Threshold'] for r in threshold_results]
accuracies = [r['Accuracy'] for r in threshold_results]
f1_scores = [r['F1-Score'] for r in threshold_results]
recalls = [r['Recall'] for r in threshold_results]
precisions = [r['Precision'] for r in threshold_results]

plt.plot(thresholds, accuracies, 'bo-', label='Accuracy', linewidth=3, markersize=8)
plt.plot(thresholds, f1_scores, 'ro-', label='F1-Score', linewidth=3, markersize=8)
plt.plot(thresholds, recalls, 'go-', label='Recall (Sensitivity)', linewidth=3, markersize=8)
plt.plot(thresholds, precisions, 'mo-', label='Precision', linewidth=3, markersize=8)

# Mark the best threshold
best_thresh = best_threshold_result['Threshold']
best_f1 = best_threshold_result['F1-Score']
plt.axvline(x=best_thresh, color='red', linestyle='--', alpha=0.7, 
            label=f'Best Threshold: {best_thresh} (F1={best_f1:.3f})')

plt.xlabel('Classification Threshold', fontsize=14)
plt.ylabel('Score', fontsize=14)
plt.title('Model Performance vs Classification Threshold\n(SVM with Grid Search Optimization)', 
          fontsize=16, fontweight='bold', pad=20)
plt.legend(fontsize=12, loc='best')
plt.grid(True, alpha=0.3)
plt.xticks(thresholds)
plt.tight_layout()
plt.savefig('threshold_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("      ✅ Saved: threshold_analysis.png")

print("\n" + "="*60)
print("GENERATING VISUALIZATIONS")
print("="*60)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Update model names to include grid search SVM
model_names = list(results.keys())
if 'SVM (Grid Search)' in model_names:
    # Move grid search SVM to the end for visualization
    model_names.remove('SVM (Grid Search)')
    model_names.append('SVM (Grid Search)')

# 6.1 Model Comparison Bar Chart (UPDATED)
print("\n   1. Creating updated model comparison chart...")
plt.figure(figsize=(14, 8))
accuracies = [results[name]['accuracy'] for name in model_names]
colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd']

bars = plt.bar(model_names, accuracies, color=colors, edgecolor='black', linewidth=2)
plt.title('Model Performance Comparison - Lung Cancer Prediction', 
          fontsize=18, fontweight='bold', pad=20)
plt.ylabel('Accuracy Score', fontsize=14)
plt.ylim(0.5, 1.0)
plt.axhline(y=max(accuracies), color='red', linestyle='--', alpha=0.7, 
            label=f'Best: {max(accuracies):.3f}')

# Add values on bars
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{acc:.3f}', ha='center', fontweight='bold', fontsize=11)

plt.legend(fontsize=12)
plt.xticks(rotation=15, fontsize=11)
plt.yticks(fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('model_comparison_updated.png', dpi=300, bbox_inches='tight')
plt.close()
print("      ✅ Saved: model_comparison_updated.png")

# 6.2 ROC Curves (UPDATED)
print("\n   2. Creating updated ROC curves...")
plt.figure(figsize=(12, 10))
for name, color in zip(model_names, colors):
    if 'y_pred_proba' in results[name]:
        fpr, tpr, _ = roc_curve(y_test, results[name]['y_pred_proba'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=3, 
                 label=f'{name} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier (AUC = 0.5)')
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('ROC Curves - Model Comparison (Including Grid Search)', 
          fontsize=18, fontweight='bold', pad=20)
plt.legend(loc='lower right', fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curves_updated.png', dpi=300, bbox_inches='tight')
plt.close()
print("      ✅ Saved: roc_curves_updated.png")

# 6.3 Confusion Matrix for Best Model (UPDATED)
best_model_name = model_names[accuracies.index(max(accuracies))]
print(f"\n   3. Creating confusion matrix for best model ({best_model_name})...")

plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, results[best_model_name]['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted No Cancer', 'Predicted Cancer'],
            yticklabels=['Actual No Cancer', 'Actual Cancer'],
            cbar_kws={'label': 'Count'}, annot_kws={"size": 14})

plt.title(f'Confusion Matrix - {best_model_name}\nAccuracy: {results[best_model_name]["accuracy"]:.3f}', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)
plt.tight_layout()
plt.savefig('confusion_matrix_updated.png', dpi=300, bbox_inches='tight')
plt.close()
print("      ✅ Saved: confusion_matrix_updated.png")

print("\n" + "="*60)
print("FINAL RESULTS SUMMARY")
print("="*60)

# 7.1 Best model details
print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   Accuracy:  {results[best_model_name]['accuracy']:.4f}")
print(f"   Precision: {results[best_model_name]['precision']:.4f}")
print(f"   Recall:    {results[best_model_name]['recall']:.4f}")
print(f"   F1-Score:  {results[best_model_name]['f1']:.4f}")

# 7.2 Complete classification report
print("\n📊 DETAILED CLASSIFICATION REPORT:")
print("-" * 50)
print(classification_report(y_test, results[best_model_name]['y_pred'],
                          target_names=['No Cancer', 'Cancer']))

# 7.3 All models comparison table (UPDATED)
print("\n📈 ALL MODELS COMPARISON (INCLUDING GRID SEARCH):")
print("-" * 70)
print(f"{'Model':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
print("-" * 70)
for name in model_names:
    print(f"{name:<25} {results[name]['accuracy']:<10.4f} {results[name]['precision']:<10.4f} "
          f"{results[name]['recall']:<10.4f} {results[name]['f1']:<10.4f}")

# 7.4 Confusion matrix breakdown
print("\n🎯 CONFUSION MATRIX BREAKDOWN:")
cm = confusion_matrix(y_test, results[best_model_name]['y_pred'])
print(f"   True Negatives (TN): {cm[0,0]} - Correctly predicted No Cancer")
print(f"   False Positives (FP): {cm[0,1]} - Predicted Cancer but was No Cancer")
print(f"   False Negatives (FN): {cm[1,0]} - Predicted No Cancer but was Cancer")
print(f"   True Positives (TP): {cm[1,1]} - Correctly predicted Cancer")

# 7.5 Performance metrics
sensitivity = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
specificity = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
print(f"\n📊 ADDITIONAL METRICS:")
print(f"   Sensitivity (Recall): {sensitivity:.4f}")
print(f"   Specificity: {specificity:.4f}")
print(f"   Balanced Accuracy: {(sensitivity + specificity)/2:.4f}")

# 7.6 Dataset statistics
print("\n📊 DATASET STATISTICS:")
print(f"   Total samples: {len(df)}")
print(f"   Training samples: {len(X_train)}")
print(f"   Testing samples: {len(X_test)}")
print(f"   Number of features: {X.shape[1]}")
print(f"   Original class imbalance: {y.mean():.1%} Cancer, {1-y.mean():.1%} No Cancer")

# 7.7 GRID SEARCH SUMMARY
print("\n🔧 GRID SEARCH OPTIMIZATION SUMMARY:")
print("-" * 50)
print(f"   Best parameters for SVM: {grid_search.best_params_}")
print(f"   Best CV F1-score: {grid_search.best_score_:.4f}")
print(f"   Test accuracy improvement: {improvement_acc:+.4f}")
print(f"   Test F1-score improvement: {improvement_f1:+.4f}")

# 7.8 THRESHOLD OPTIMIZATION SUMMARY
print("\n🎯 THRESHOLD OPTIMIZATION SUMMARY:")
print("-" * 50)
print(f"   Best classification threshold: {best_threshold_result['Threshold']}")
print(f"   Best F1-score at this threshold: {best_threshold_result['F1-Score']:.4f}")
print(f"   Corresponding accuracy: {best_threshold_result['Accuracy']:.4f}")
print(f"   Corresponding recall: {best_threshold_result['Recall']:.4f}")

print("\n" + "="*60)
print("✅ TRAINING AND EVALUATION COMPLETE!")
print("="*60)
print("\n📁 FILES GENERATED:")
print("   1. model_comparison_updated.png - Updated accuracy comparison")
print("   2. roc_curves_updated.png - Updated ROC curves")
print("   3. confusion_matrix_updated.png - Confusion matrix for best model")
print("   4. threshold_analysis.png - Threshold optimization analysis")
if best_model_name in ['Random Forest', 'XGBoost']:
    print("   5. feature_importance.png - Top 15 important features")
elif best_model_name == 'Logistic Regression':
    print("   5. feature_coefficients.png - Top 15 feature coefficients")

print(f"\n🎯 RECOMMENDATION: Use {best_model_name} with threshold {best_threshold_result['Threshold']}")
print("   for lung cancer prediction with the provided model and scaler.")