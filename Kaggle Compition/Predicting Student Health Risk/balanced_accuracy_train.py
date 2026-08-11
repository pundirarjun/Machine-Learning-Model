"""
Student Health Risk Prediction - Balanced Accuracy Optimized Training
Metric: Balanced Accuracy
Strategy: Multiple models + target encoding + stratified CV + ensemble
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from scipy.stats import mode as scipy_mode

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

warnings.filterwarnings('ignore')

RANDOM_STATE = 42
TARGET = 'health_condition'
ID_COL = 'id'
N_FOLDS = 5

ROOT = Path(__file__).parent
OUTPUT_DIR = ROOT / 'submissions'
TRAIN_PATH = ROOT / 'train.csv'
TEST_PATH = ROOT / 'test.csv'
SAMPLE_SUBMISSION_PATH = ROOT / 'sample_submission.csv'

def safe_divide(numerator, denominator, fill_value=np.nan):
    denominator = denominator.replace(0, np.nan) if isinstance(denominator, pd.Series) else denominator
    result = numerator / denominator
    return result.replace([np.inf, -np.inf], np.nan).fillna(fill_value)

def add_features(df):
    out = df.copy()
    numeric_base = ['sleep_duration', 'heart_rate', 'bmi', 'calorie_expenditure', 'step_count', 'exercise_duration', 'water_intake']
    available_numeric = [col for col in numeric_base if col in out.columns]
    out['missing_feature_count'] = out[available_numeric].isna().sum(axis=1)
    out['steps_per_exercise_min'] = safe_divide(out['step_count'], out['exercise_duration'] + 1)
    out['calories_per_step'] = safe_divide(out['calorie_expenditure'], out['step_count'] + 1)
    out['calories_per_exercise_min'] = safe_divide(out['calorie_expenditure'], out['exercise_duration'] + 1)
    out['water_per_bmi'] = safe_divide(out['water_intake'], out['bmi'])
    out['sleep_bmi_interaction'] = out['sleep_duration'] * out['bmi']
    out['heart_rate_bmi_interaction'] = out['heart_rate'] * out['bmi']
    out['sleep_deficit_from_8h'] = (8 - out['sleep_duration']).clip(lower=0)
    out['sleep_excess_over_9h'] = (out['sleep_duration'] - 9).clip(lower=0)
    out['activity_score'] = (
        safe_divide(out['step_count'], pd.Series(10000, index=out.index)) +
        safe_divide(out['exercise_duration'], pd.Series(60, index=out.index)) +
        safe_divide(out['calorie_expenditure'], pd.Series(2500, index=out.index)) +
        safe_divide(out['water_intake'], pd.Series(2, index=out.index))
    )
    out['bmi_category'] = pd.cut(out['bmi'], bins=[-np.inf, 18.5, 25, 30, np.inf], labels=['underweight', 'normal', 'overweight', 'obese']).astype('object')
    out['sleep_duration_bin'] = pd.cut(out['sleep_duration'], bins=[-np.inf, 5, 7, 9, np.inf], labels=['very_short', 'short', 'recommended', 'long']).astype('object')
    out['step_count_bin'] = pd.cut(out['step_count'], bins=[-np.inf, 5000, 10000, 15000, np.inf], labels=['low', 'moderate', 'high', 'very_high']).astype('object')
    out['water_intake_bin'] = pd.cut(out['water_intake'], bins=[-np.inf, 1.5, 2.5, 3.5, np.inf], labels=['low', 'moderate', 'high', 'very_high']).astype('object')
    out['heart_rate_bin'] = pd.cut(out['heart_rate'], bins=[-np.inf, 60, 80, 100, np.inf], labels=['low', 'normal', 'elevated', 'high']).astype('object')
    return out

# Load data
print('Loading data...')
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)

train_fe = add_features(train_df)
test_fe = add_features(test_df)

feature_cols = [c for c in train_fe.columns if c not in {ID_COL, TARGET}]
numeric_cols = train_fe[feature_cols].select_dtypes(include=['number', 'bool']).columns.tolist()
categorical_cols = [c for c in feature_cols if c not in numeric_cols]

# Encode target
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(train_fe[TARGET])
X = train_fe.drop(columns=[TARGET])
X_test = test_fe.copy()

classes = label_encoder.classes_
n_classes = len(classes)
print(f'Train: {X.shape[0]}, Test: {X_test.shape[0]}')
print(f'Classes: {classes}')
counts = pd.Series(y).value_counts().sort_index()
for i, c in enumerate(classes):
    print(f'  {c}: {counts[i]} ({counts[i]/len(y)*100:.2f}%)')

# Build preprocessor
preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
        ('scaler', StandardScaler()),
    ]), numeric_cols),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
    ]), categorical_cols),
])

X_processed = preprocessor.fit_transform(X)
X_test_processed = preprocessor.transform(X_test)

# ============== MODEL DEFINITIONS ==============

# Balanced sample weights
sample_weights = compute_sample_weight('balanced', y)

# XGBoost configs
xgb_configs = [
    ('xgb_d6_850_lr04_a085', {'n_estimators': 850, 'max_depth': 6, 'learning_rate': 0.04, 'reg_alpha': 0.85, 'reg_lambda': 1.0, 'subsample': 0.8, 'colsample_bytree': 0.8}),
    ('xgb_d7_1000_lr035_a09', {'n_estimators': 1000, 'max_depth': 7, 'learning_rate': 0.035, 'reg_alpha': 0.9, 'reg_lambda': 1.5, 'subsample': 0.8, 'colsample_bytree': 0.85}),
    ('xgb_d5_750_lr04_a095', {'n_estimators': 750, 'max_depth': 5, 'learning_rate': 0.04, 'reg_alpha': 0.95, 'reg_lambda': 2.0, 'subsample': 0.75, 'colsample_bytree': 0.75}),
    ('xgb_d6_1000_lr035_a09', {'n_estimators': 1000, 'max_depth': 6, 'learning_rate': 0.035, 'reg_alpha': 0.9, 'reg_lambda': 1.0, 'subsample': 0.85, 'colsample_bytree': 0.85}),
    ('xgb_d7_750_lr035_a09', {'n_estimators': 750, 'max_depth': 7, 'learning_rate': 0.035, 'reg_alpha': 0.9, 'reg_lambda': 2.0, 'subsample': 0.8, 'colsample_bytree': 0.8}),
]

# LightGBM configs
lgb_configs = [
    ('lgb_d6_850_lr04', {'n_estimators': 850, 'max_depth': 6, 'learning_rate': 0.04, 'reg_alpha': 0.85, 'reg_lambda': 1.0, 'subsample': 0.8, 'colsample_bytree': 0.8, 'class_weight': 'balanced'}),
    ('lgb_d8_1000_lr035', {'n_estimators': 1000, 'max_depth': 8, 'learning_rate': 0.035, 'reg_alpha': 0.9, 'reg_lambda': 1.5, 'subsample': 0.8, 'colsample_bytree': 0.85, 'class_weight': 'balanced'}),
    ('lgb_d5_750_lr05', {'n_estimators': 750, 'max_depth': 5, 'learning_rate': 0.05, 'reg_alpha': 0.95, 'reg_lambda': 2.0, 'subsample': 0.75, 'colsample_bytree': 0.75, 'class_weight': 'balanced'}),
]

# CatBoost configs
cb_configs = [
    ('cb_d6_850_lr04', {'iterations': 850, 'depth': 6, 'learning_rate': 0.04, 'l2_leaf_reg': 3.0, 'auto_class_weights': 'Balanced'}),
    ('cb_d8_1000_lr035', {'iterations': 1000, 'depth': 8, 'learning_rate': 0.035, 'l2_leaf_reg': 5.0, 'auto_class_weights': 'Balanced'}),
]

# ============== STRATIFIED K-FOLD TRAINING ==============
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

all_oof_preds = {}
all_test_probas = {}

print(f'\n{"="*60}')
print('STRATIFIED K-FOLD TRAINING')
print(f'{"="*60}')

# Process each model group
def train_model_group(name_prefix, configs, model_type, use_balanced=True):
    oof_preds = np.zeros((len(y), n_classes))
    test_probas = np.zeros((len(X_test_processed), n_classes))
    oof_scores = []
    
    for name_suffix, params in configs:
        full_name = f'{name_prefix}_{name_suffix}'
        print(f'\n--- {full_name} ---')
        
        fold_preds = np.zeros((len(X_test_processed), n_classes))
        fold_oof = np.zeros((len(y), n_classes))
        fold_oof_counts = np.zeros(len(y))
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_processed, y)):
            X_tr_f, X_val_f = X_processed[train_idx], X_processed[val_idx]
            y_tr_f, y_val_f = y[train_idx], y[val_idx]
            
            sw = sample_weights[train_idx] if use_balanced else None
            
            if model_type == 'xgb':
                model = xgb.XGBClassifier(**params, eval_metric='mlogloss', random_state=RANDOM_STATE, n_jobs=-1)
                model.fit(X_tr_f, y_tr_f, sample_weight=sw, verbose=0)
            elif model_type == 'lgb':
                model = lgb.LGBMClassifier(**params, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
                model.fit(X_tr_f, y_tr_f, sample_weight=sw)
            elif model_type == 'cb':
                model = cb.CatBoostClassifier(**params, random_seed=RANDOM_STATE, verbose=0)
                model.fit(X_tr_f, y_tr_f, sample_weight=sw)
            
            # OOF predictions
            probas = model.predict_proba(X_val_f)
            for i, idx in enumerate(val_idx):
                fold_oof[idx] += probas[i]
                fold_oof_counts[idx] += 1
            
            # Test predictions
            test_proba = model.predict_proba(X_test_processed)
            fold_preds += test_proba / N_FOLDS
            
            # Balanced accuracy
            val_pred = model.predict(X_val_f)
            bal_acc = balanced_accuracy_score(y_val_f, val_pred)
            cv_scores.append(bal_acc)
            print(f'  Fold {fold+1}: balanced_acc = {bal_acc:.5f}')
        
        # Average OOF
        fold_oof = fold_oof / np.maximum(fold_oof_counts[:, None], 1)
        oof_pred = np.argmax(fold_oof, axis=1)
        oof_bal_acc = balanced_accuracy_score(y, oof_pred)
        
        print(f'  OOF balanced_acc: {oof_bal_acc:.5f}')
        print(f'  Mean CV: {np.mean(cv_scores):.5f} (+/- {np.std(cv_scores):.5f})')
        
        all_oof_preds[full_name] = oof_pred
        test_probas += fold_preds
        oof_scores.append((full_name, oof_bal_acc, np.mean(cv_scores)))
    
    return oof_scores, test_probas

print('\n=== XGBoost Models ===')
xgb_scores, xgb_test_probas = train_model_group('xgb', xgb_configs, 'xgb', use_balanced=True)

print('\n=== LightGBM Models ===')
lgb_scores, lgb_test_probas = train_model_group('lgb', lgb_configs, 'lgb', use_balanced=True)

print('\n=== CatBoost Models ===')
cb_scores, cb_test_probas = train_model_group('cb', cb_configs, 'cb', use_balanced=True)

# ============== EVALUATE ALL MODELS ==============
print(f'\n{"="*60}')
print('ALL MODEL SCORES (Balanced Accuracy)')
print(f'{"="*60}')

all_model_scores = xgb_scores + lgb_scores + cb_scores
all_model_scores.sort(key=lambda x: x[1], reverse=True)

for name, oof_bal, cv_mean in all_model_scores:
    print(f'  {name:35s}: OOF={oof_bal:.5f} | CV={cv_mean:.5f}')

best_name = all_model_scores[0][0]
best_oof = all_model_scores[0][1]
print(f'\nBest single model: {best_name} (OOF={best_oof:.5f})')

# ============== ENSEMBLE ==============
print(f'\n{"="*60}')
print('ENSEMBLE CREATION')
print(f'{"="*60}')

# Collect test predictions from all models and ensemble
all_test_preds = []

for name, oof_bal, cv_mean in all_model_scores:
    preds = all_oof_preds[name]
    all_test_preds.append(preds)

    # Also get avg proba predictions for soft voting
    # (already accumulated in test_probas)

# Average all test probabilities
n_models = len(all_model_scores)
avg_probas = np.zeros((len(X_test_processed), n_classes))
idx = 0

# Average XGB test probas
avg_probas += xgb_test_probas

# Average LGB test probas
avg_probas += lgb_test_probas

# Average CB test probas
avg_probas += cb_test_probas

avg_probas /= 3  # Average of model groups

# Soft voting from averaged probabilities
soft_pred = np.argmax(avg_probas, axis=1)
soft_labels = label_encoder.inverse_transform(soft_pred)

# Hard voting across all individual models
all_model_preds = []
for name, _, _ in all_model_scores:
    oof_pred = all_oof_preds[name]
    # Refit on full data to get test predictions
    print(f'Fitting {name} on full data for ensemble...')
    
    # Find config
    for prefix, configs, mtype, _ in [
        ('xgb', xgb_configs, 'xgb', True),
        ('lgb', lgb_configs, 'lgb', True),
        ('cb', cb_configs, 'cb', True),
    ]:
        for suffix, params in configs:
            full = f'{prefix}_{suffix}'
            if full == name:
                if mtype == 'xgb':
                    model = xgb.XGBClassifier(**params, random_state=RANDOM_STATE, n_jobs=-1, eval_metric='mlogloss')
                elif mtype == 'lgb':
                    model = lgb.LGBMClassifier(**params, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
                elif mtype == 'cb':
                    model = cb.CatBoostClassifier(**params, random_seed=RANDOM_STATE, verbose=0)
                
                model.fit(X_processed, y, sample_weight=sample_weights)
                test_pred = model.predict(X_test_processed)
                all_model_preds.append(test_pred)
                break

# Hard voting ensemble
hard_vote, _ = scipy_mode(np.column_stack(all_model_preds), axis=1)
hard_vote = hard_vote.flatten().astype(int)
hard_labels = label_encoder.inverse_transform(hard_vote)

# ============== GENERATE SUBMISSIONS ==============
print(f'\n{"="*60}')
print('GENERATING SUBMISSIONS')
print(f'{"="*60}')

# 1. Best single model submission
best_config = all_model_scores[0][0]
print(f'\n1. Best single model: {best_config}')

for prefix, configs, mtype, _ in [
    ('xgb', xgb_configs, 'xgb', True),
    ('lgb', lgb_configs, 'lgb', True),
    ('cb', cb_configs, 'cb', True),
]:
    for suffix, params in configs:
        full = f'{prefix}_{suffix}'
        if full == best_config:
            if mtype == 'xgb':
                final_model = xgb.XGBClassifier(**params, random_state=RANDOM_STATE, n_jobs=-1, eval_metric='mlogloss')
            elif mtype == 'lgb':
                final_model = lgb.LGBMClassifier(**params, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
            elif mtype == 'cb':
                final_model = cb.CatBoostClassifier(**params, random_seed=RANDOM_STATE, verbose=0)
            
            final_model.fit(X_processed, y, sample_weight=sample_weights)
            test_pred = final_model.predict(X_test_processed)
            test_labels = label_encoder.inverse_transform(test_pred)
            
            sub = sample_submission.copy()
            sub[TARGET] = test_labels
            p = OUTPUT_DIR / f'submission_{best_config}_best.csv'
            sub.to_csv(p, index=False)
            print(f'  Saved: {p.name}')
            print(f'  Dist: {pd.Series(test_labels).value_counts().to_dict()}')
            break

# 2. Soft voting ensemble
print('\n2. Soft Voting Ensemble')
sub = sample_submission.copy()
sub[TARGET] = soft_labels
p = OUTPUT_DIR / 'submission_soft_vote_ensemble.csv'
sub.to_csv(p, index=False)
print(f'  Saved: {p.name}')
print(f'  Dist: {pd.Series(soft_labels).value_counts().to_dict()}')

# 3. Hard voting ensemble
print('\n3. Hard Voting Ensemble')
sub = sample_submission.copy()
sub[TARGET] = hard_labels
p = OUTPUT_DIR / 'submission_hard_vote_ensemble.csv'
sub.to_csv(p, index=False)
print(f'  Saved: {p.name}')
print(f'  Dist: {pd.Series(hard_labels).value_counts().to_dict()}')

# 4. Best XGBoost model retrained on full data
print('\n4. XGBoost best single full data retrain')
# Use the best XGBoost config
xgb_scores_sorted = sorted(xgb_scores, key=lambda x: x[1], reverse=True)
best_xgb_name = xgb_scores_sorted[0][0]
for suffix, params in xgb_configs:
    full = f'xgb_{suffix}'
    if full == best_xgb_name:
        model = xgb.XGBClassifier(**params, random_state=RANDOM_STATE, n_jobs=-1, eval_metric='mlogloss')
        model.fit(X_processed, y, sample_weight=sample_weights)
        test_pred = model.predict(X_test_processed)
        test_labels = label_encoder.inverse_transform(test_pred)
        sub = sample_submission.copy()
        sub[TARGET] = test_labels
        p = OUTPUT_DIR / f'submission_{best_xgb_name}_full.csv'
        sub.to_csv(p, index=False)
        print(f'  Saved: {p.name}')
        print(f'  Dist: {pd.Series(test_labels).value_counts().to_dict()}')
        break

print(f'\n{"="*60}')
print('DONE - All submissions saved')
print(f'{"="*60}')