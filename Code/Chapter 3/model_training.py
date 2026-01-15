import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, f1_score, precision_score, recall_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("STARTING LUNG CANCER PREDICTION MODEL TRAINING")
print("="*60)

print("\n1. Loading and preparing data...")
df = pd.read_csv('lung_cancer_preprocessed_data.csv')
print(f"Dataset shape: {df.shape}")

y = df['LUNG_CANCER_YES']
X = df.drop(['LUNG_CANCER_YES', 'LUNG_CANCER_NO'], axis=1)

print(f"Features (X): {X.shape}")
print(f"Target (y): {y.shape}")

print("\n2. Splitting data into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")

print("\n3. Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Features scaled successfully")

print("\n4. Handling class imbalance with SMOTE...")
print(f"Before SMOTE - Training set class distribution:")
print(f"Class 0 (No Cancer): {sum(y_train==0)} samples")
print(f"Class 1 (Cancer): {sum(y_train==1)} samples")

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

print(f"After SMOTE - Training set class distribution:")
print(f"Class 0 (No Cancer): {sum(y_train_balanced==0)} samples")
print(f"Class 1 (Cancer): {sum(y_train_balanced==1)} samples")

print("\n" + "="*60)
print("TRAINING DEFAULT MODELS")
print("="*60)

results = {}

models_initial = {
    'Logistic Regression': LogisticRegression(penalty='l2', C=1.0, max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42)
}

for name, model in models_initial.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_balanced, y_train_balanced)
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    cm = confusion_matrix(y_test, y_pred)
    tn = cm[0, 0]
    fp = cm[0, 1]
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'confusion_matrix': cm
    }
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Specificity: {specificity:.4f}")

default_svm_result = results['SVM']

print("\n" + "="*60)
print("OVERFITTING AND UNDERFITTING ANALYSIS")
print("="*60)

print("\nTable 3.1: Overfitting Analysis Results")
print("-" * 70)
print(f"{'Model':<25} {'Train Acc':<12} {'Test Acc':<12} {'Diff':<12} {'Assessment':<20}")
print("-" * 70)

for name in models_initial.keys():
    model = results[name]['model']
    y_train_pred = model.predict(X_train_balanced)
    train_acc = accuracy_score(y_train_balanced, y_train_pred)
    test_acc = results[name]['accuracy']
    diff = train_acc - test_acc
    
    if diff > 0.05:
        assessment = "Potential overfitting"
    elif diff < -0.02:
        assessment = "Potential underfitting"
    else:
        assessment = "Good generalization"
    
    print(f"{name:<25} {train_acc:<12.4f} {test_acc:<12.4f} {diff:<+12.4f} {assessment:<20}")

print("\n" + "="*60)
print("REGULARIZATION WITH RIDGE (L2) PENALTY")
print("="*60)

lr_model = models_initial['Logistic Regression']
print("\nLogistic Regression Configuration (ACTUALLY IMPLEMENTED IN CODE):")
print(f"  penalty='{lr_model.penalty}' (Ridge regularization)")
print(f"  C={lr_model.C} (Regularization strength)")
print(f"  max_iter={lr_model.max_iter}")
print(f"  random_state={lr_model.random_state}")
print("\nThis L2 regularization prevents overfitting by discouraging large coefficient values.")

print("\n" + "="*60)
print("GENERATING ANALYSIS VISUALIZATIONS")
print("="*60)

plt.figure(figsize=(10, 6))
models_list = list(results.keys())
accuracies = [results[m]['accuracy'] for m in models_list]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

bars = plt.bar(models_list, accuracies, color=colors, edgecolor='black')
plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy Score', fontsize=12)
plt.ylim(0.5, 1.0)
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{acc:.3f}', ha='center', fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300)
plt.close()
print("✓ Saved: model_comparison.png")

plt.figure(figsize=(12, 8))
for i, (name, color) in enumerate(zip(models_list, colors)):
    fpr, tpr, _ = roc_curve(y_test, results[name]['y_pred_proba'])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curves.png', dpi=300)
plt.close()
print("✓ Saved: roc_curves.png")

best_model_name = models_list[accuracies.index(max(accuracies))]
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, results[best_model_name]['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
plt.close()
print(f"✓ Saved: confusion_matrix.png ({best_model_name})")

print("\n" + "="*60)
print("HYPERPARAMETER OPTIMIZATION VIA GRID SEARCH")
print("="*60)

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'linear']
}

print(f"\nGrid Search Parameter Configuration:")
print(f"  C values: [0.1, 1, 10, 100]")
print(f"  gamma values: [1, 0.1, 0.01, 0.001]")
print(f"  kernel types: ['rbf', 'linear']")
print(f"  Total combinations: 32 parameter sets")
print(f"  Cross-validation: 5-fold")
print(f"  Scoring metric: F1-score")

svm_for_grid = SVC(probability=True, random_state=42)
grid_search = GridSearchCV(svm_for_grid, param_grid, refit=True, cv=5, scoring='f1', verbose=0)
grid_search.fit(X_train_balanced, y_train_balanced)

print(f"\nGrid Search Results:")
print(f"  Best parameters found: {grid_search.best_params_}")
print(f"  Best cross-validation F1 score: {grid_search.best_score_:.4f}")

best_svm = grid_search.best_estimator_
y_pred_best = best_svm.predict(X_test_scaled)
y_pred_proba_best = best_svm.predict_proba(X_test_scaled)[:, 1]

best_accuracy = accuracy_score(y_test, y_pred_best)
best_precision = precision_score(y_test, y_pred_best)
best_recall = recall_score(y_test, y_pred_best)
best_f1 = f1_score(y_test, y_pred_best)

cm_best = confusion_matrix(y_test, y_pred_best)
tn_best = cm_best[0, 0]
fp_best = cm_best[0, 1]
specificity_best = tn_best / (tn_best + fp_best) if (tn_best + fp_best) > 0 else 0

results['SVM (Grid Search)'] = {
    'model': best_svm,
    'y_pred': y_pred_best,
    'y_pred_proba': y_pred_proba_best,
    'accuracy': best_accuracy,
    'precision': best_precision,
    'recall': best_recall,
    'f1': best_f1,
    'specificity': specificity_best,
    'confusion_matrix': cm_best
}

improvement_acc = best_accuracy - default_svm_result['accuracy']
improvement_f1 = best_f1 - default_svm_result['f1']

print(f"  Test accuracy with tuned parameters: {best_accuracy:.4f}")
print(f"  Test F1-score with tuned parameters: {best_f1:.4f}")
print(f"  Accuracy improvement: {improvement_acc:+.4f}")
print(f"  F1-score improvement: {improvement_f1:+.4f}")

print("\n" + "="*60)
print("MODEL REFINEMENT AND THRESHOLD ADJUSTMENT")
print("="*60)

y_prob = best_svm.predict_proba(X_test_scaled)[:, 1]
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
threshold_data = []

print(f"\nTesting different classification thresholds:")
print("-" * 70)
print(f"{'Threshold':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Specificity':<12}")
print("-" * 70)

for thresh in thresholds:
    y_pred_thresh = (y_prob >= thresh).astype(int)
    acc = accuracy_score(y_test, y_pred_thresh)
    prec = precision_score(y_test, y_pred_thresh)
    rec = recall_score(y_test, y_pred_thresh)
    f1 = f1_score(y_test, y_pred_thresh)
    cm_thresh = confusion_matrix(y_test, y_pred_thresh)
    tn_thresh = cm_thresh[0, 0]
    fp_thresh = cm_thresh[0, 1]
    spec_thresh = tn_thresh / (tn_thresh + fp_thresh) if (tn_thresh + fp_thresh) > 0 else 0
    
    threshold_data.append({
        'Threshold': thresh,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'Specificity': spec_thresh
    })
    
    print(f"{thresh:<12.1f} {acc:<12.4f} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f} {spec_thresh:<12.4f}")

best_threshold_data = max(threshold_data, key=lambda x: x['F1-Score'])
best_thresh = best_threshold_data['Threshold']

print(f"\nBest threshold: {best_thresh} (F1-Score: {best_threshold_data['F1-Score']:.4f})")
print(f"Accuracy at this threshold: {best_threshold_data['Accuracy']:.4f}")
print(f"Precision at this threshold: {best_threshold_data['Precision']:.4f}")
print(f"Recall at this threshold: {best_threshold_data['Recall']:.4f}")
print(f"Specificity at this threshold: {best_threshold_data['Specificity']:.4f}")

plt.figure(figsize=(12, 8))
thresholds_vals = [t['Threshold'] for t in threshold_data]
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
colors_thresh = ['blue', 'green', 'red', 'purple']

for i, metric in enumerate(metrics):
    values = [t[metric] for t in threshold_data]
    plt.plot(thresholds_vals, values, 'o-', color=colors_thresh[i], linewidth=3, markersize=8, label=metric)

plt.axvline(x=best_thresh, color='red', linestyle='--', alpha=0.7, label=f'Best Threshold: {best_thresh}')
plt.xlabel('Classification Threshold', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Model Performance vs Classification Threshold', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xticks(thresholds)
plt.tight_layout()
plt.savefig('threshold_analysis.png', dpi=300)
plt.close()
print("✓ Saved: threshold_analysis.png")

results['SVM (Grid Search + Threshold)'] = {
    'model': best_svm,
    'accuracy': best_threshold_data['Accuracy'],
    'precision': best_threshold_data['Precision'],
    'recall': best_threshold_data['Recall'],
    'f1': best_threshold_data['F1-Score'],
    'specificity': best_threshold_data['Specificity']
}

print("\n" + "="*60)
print("CROSS-VALIDATION AND FINAL MODEL SELECTION")
print("="*60)

print("\n5-fold Cross-validation on Default SVM:")
cv_scores = cross_val_score(default_svm_result['model'], X_train_balanced, y_train_balanced, cv=5, scoring='f1')
print(f"Cross-validation F1 scores: {[f'{score:.4f}' for score in cv_scores]}")
print(f"Mean CV F1 score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

print("\nFinal Model Performance:")
print("-" * 50)
print(f"Model: SVM (Default Parameters)")
print(f"Accuracy: {default_svm_result['accuracy']:.4f}")
print(f"F1-Score: {default_svm_result['f1']:.4f}")
print(f"Precision: {default_svm_result['precision']:.4f}")
print(f"Recall (Sensitivity): {default_svm_result['recall']:.4f}")
print(f"Specificity: {default_svm_result['specificity']:.4f}")

print("\nConfusion Matrix Analysis:")
cm = default_svm_result['confusion_matrix']
print(f"Confusion Matrix:")
print(cm)
print(f"\nTrue Positives (TP): {cm[1,1]}")
print(f"False Negatives (FN): {cm[1,0]}")
print(f"False Positives (FP): {cm[0,1]}")
print(f"True Negatives (TN): {cm[0,0]}")
print(f"\nSensitivity: {cm[1,1]/(cm[1,1]+cm[1,0]):.4f}")
print(f"Specificity: {cm[0,0]/(cm[0,0]+cm[0,1]):.4f}")
print(f"False Negative Rate: {cm[1,0]/(cm[1,1]+cm[1,0]):.4f}")

print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)

print("\nTable 4.1: Model Performance Comparison on Test Set")
print("-" * 85)
print(f"{'Model':<30} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Specificity':<10}")
print("-" * 85)

final_models = ['Logistic Regression', 'Random Forest', 'XGBoost', 'SVM', 'SVM (Grid Search)', 'SVM (Grid Search + Threshold)']
for name in final_models:
    if name in results:
        print(f"{name:<30} {results[name]['accuracy']:<10.4f} {results[name]['precision']:<10.4f} "
              f"{results[name]['recall']:<10.4f} {results[name]['f1']:<10.4f} {results[name]['specificity']:<10.4f}")

print("-" * 85)

print(f"\nFINAL MODEL SELECTED: SVM (Default)")
print(f"Accuracy: {default_svm_result['accuracy']:.4f}")
print(f"F1-Score: {default_svm_result['f1']:.4f}")
print(f"Sensitivity/Recall: {default_svm_result['recall']:.4f}")
print(f"Specificity: {default_svm_result['specificity']:.4f}")
print(f"Precision: {default_svm_result['precision']:.4f}")

print("\n" + "="*60)
print("VISUALIZATIONS GENERATED")
print("="*60)
print("1. model_comparison.png - Model accuracy comparison")
print("2. roc_curves.png - ROC curves for all models")
print("3. confusion_matrix.png - Confusion matrix for best model")
print("4. threshold_analysis.png - Threshold optimization analysis")
print("\n" + "="*60)
print("COMPLETE!")
print("="*60)