"""
Improved pipeline for Kaggle: adds LightGBM and XGBoost, performs StratifiedKFold CV with out-of-fold probability predictions,
optional RandomizedSearchCV tuning for LGBM/XGB, and creates ensembled submission CSV(s).

Usage:
  python improved_pipeline.py --tune True --n_folds 5 --n_iter 12

Notes:
- This script re-implements the feature engineering from main.ipynb. It expects train.csv, test.csv, sample_submission.csv
  in the same working directory.
- RandomizedSearchCV can be slow; default n_iter is conservative. Increase n_iter if you have time and resources.
"""

import argparse
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

RANDOM_STATE = 42
TARGET = 'health_condition'
ID_COL = 'id'

ROOT = Path.cwd()
TRAIN_PATH = ROOT / 'train.csv'
TEST_PATH = ROOT / 'test.csv'
SAMPLE_SUBMISSION_PATH = ROOT / 'sample_submission.csv'
OUTPUT_DIR = ROOT / 'submissions'
OUTPUT_DIR.mkdir(exist_ok=True)

warnings.filterwarnings('ignore', category=FutureWarning)


# --- Feature engineering (adapted from notebook) ---

def safe_divide(numerator, denominator, fill_value=np.nan):
    denominator = denominator.replace(0, np.nan) if isinstance(denominator, pd.Series) else denominator
    result = numerator / denominator
    return result.replace([np.inf, -np.inf], np.nan).fillna(fill_value)


def add_features(df):
    out = df.copy()
    numeric_base = [
        'sleep_duration', 'heart_rate', 'bmi', 'calorie_expenditure',
        'step_count', 'exercise_duration', 'water_intake'
    ]
    available_numeric = [col for col in numeric_base if col in out.columns]
    out['missing_feature_count'] = out[available_numeric].isna().sum(axis=1)

    # safe engineered features
    out['steps_per_exercise_min'] = safe_divide(out.get('step_count', pd.Series(0, index=out.index)), out.get('exercise_duration', pd.Series(0, index=out.index)) + 1)
    out['calories_per_step'] = safe_divide(out.get('calorie_expenditure', pd.Series(0, index=out.index)), out.get('step_count', pd.Series(0, index=out.index)) + 1)
    out['calories_per_exercise_min'] = safe_divide(out.get('calorie_expenditure', pd.Series(0, index=out.index)), out.get('exercise_duration', pd.Series(0, index=out.index)) + 1)
    out['water_per_bmi'] = safe_divide(out.get('water_intake', pd.Series(0, index=out.index)), out.get('bmi', pd.Series(np.nan, index=out.index)))
    out['sleep_bmi_interaction'] = out.get('sleep_duration', pd.Series(np.nan, index=out.index)) * out.get('bmi', pd.Series(np.nan, index=out.index))
    out['heart_rate_bmi_interaction'] = out.get('heart_rate', pd.Series(np.nan, index=out.index)) * out.get('bmi', pd.Series(np.nan, index=out.index))
    out['sleep_deficit_from_8h'] = (8 - out.get('sleep_duration', pd.Series(np.nan, index=out.index))).clip(lower=0)
    out['sleep_excess_over_9h'] = (out.get('sleep_duration', pd.Series(np.nan, index=out.index)) - 9).clip(lower=0)
    out['activity_score'] = (
        safe_divide(out.get('step_count', pd.Series(0, index=out.index)), pd.Series(10000, index=out.index))
        + safe_divide(out.get('exercise_duration', pd.Series(0, index=out.index)), pd.Series(60, index=out.index))
        + safe_divide(out.get('calorie_expenditure', pd.Series(0, index=out.index)), pd.Series(2500, index=out.index))
        + safe_divide(out.get('water_intake', pd.Series(0, index=out.index)), pd.Series(2, index=out.index))
    )

    # binned categories
    if 'bmi' in out.columns:
        out['bmi_category'] = pd.cut(
            out['bmi'],
            bins=[-np.inf, 18.5, 25, 30, np.inf],
            labels=['underweight', 'normal', 'overweight', 'obese'],
        ).astype('object')

    if 'sleep_duration' in out.columns:
        out['sleep_duration_bin'] = pd.cut(
            out['sleep_duration'],
            bins=[-np.inf, 5, 7, 9, np.inf],
            labels=['very_short', 'short', 'recommended', 'long'],
        ).astype('object')

    if 'step_count' in out.columns:
        out['step_count_bin'] = pd.cut(
            out['step_count'],
            bins=[-np.inf, 5000, 10000, 15000, np.inf],
            labels=['low', 'moderate', 'high', 'very_high'],
        ).astype('object')

    if 'water_intake' in out.columns:
        out['water_intake_bin'] = pd.cut(
            out['water_intake'],
            bins=[-np.inf, 1.5, 2.5, 3.5, np.inf],
            labels=['low', 'moderate', 'high', 'very_high'],
        ).astype('object')

    if 'heart_rate' in out.columns:
        out['heart_rate_bin'] = pd.cut(
            out['heart_rate'],
            bins=[-np.inf, 60, 80, 100, np.inf],
            labels=['low', 'normal', 'elevated', 'high'],
        ).astype('object')

    return out


# --- Helper: split columns and preprocessor builders ---

def split_columns(df):
    feature_columns = [col for col in df.columns if col not in {ID_COL, TARGET}]
    numeric_columns = df[feature_columns].select_dtypes(include=['number', 'bool']).columns.tolist()
    categorical_columns = [col for col in feature_columns if col not in numeric_columns]
    return numeric_columns, categorical_columns


def make_onehot_preprocessor(numeric_columns, categorical_columns, scale_numeric):
    numeric_steps = [('imputer', SimpleImputer(strategy='median', add_indicator=True))]
    if scale_numeric:
        numeric_steps.append(('scaler', StandardScaler()))

    categorical_pipeline = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ('num', Pipeline(numeric_steps), numeric_columns),
            ('cat', categorical_pipeline, categorical_columns),
        ]
    )


def make_ordinal_preprocessor(numeric_columns, categorical_columns):
    return ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median', add_indicator=True), numeric_columns),
            (
                'cat',
                Pipeline(
                    steps=[
                        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )


# --- Main execution ---

def main(tune=False, n_folds=5, n_iter=12):
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)

    train_fe = add_features(train_df)
    test_fe = add_features(test_df)

    X = train_fe.drop(columns=[TARGET])
    y = train_fe[TARGET]
    X_test = test_fe.copy()

    numeric_cols, categorical_cols = split_columns(train_fe)

    linear_preprocessor = make_onehot_preprocessor(numeric_cols, categorical_cols, scale_numeric=True)
    tree_preprocessor = make_onehot_preprocessor(numeric_cols, categorical_cols, scale_numeric=False)
    ordinal_preprocessor = make_ordinal_preprocessor(numeric_cols, categorical_cols)

    # Define models (defaults). We'll tune LGBM/XGB if requested.
    models = {
        'lgbm': Pipeline(steps=[('preprocess', tree_preprocessor), ('model', LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31, objective='multiclass', random_state=RANDOM_STATE, n_jobs=-1))]),
        'xgb': Pipeline(steps=[('preprocess', tree_preprocessor), ('model', XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=6, objective='multi:softprob', use_label_encoder=False, eval_metric='mlogloss', random_state=RANDOM_STATE, n_jobs=-1))]),
        'random_forest': Pipeline(steps=[('preprocess', tree_preprocessor), ('model', RandomForestClassifier(n_estimators=120, max_depth=18, min_samples_leaf=10, class_weight='balanced_subsample', n_jobs=-1, random_state=RANDOM_STATE))]),
        'extra_trees': Pipeline(steps=[('preprocess', tree_preprocessor), ('model', ExtraTreesClassifier(n_estimators=160, max_depth=22, min_samples_leaf=8, class_weight='balanced', n_jobs=-1, random_state=RANDOM_STATE))]),
        'hist_gb': Pipeline(steps=[('preprocess', ordinal_preprocessor), ('model', HistGradientBoostingClassifier(learning_rate=0.06, max_iter=220, max_leaf_nodes=31, l2_regularization=0.05, early_stopping=True, random_state=RANDOM_STATE))]),
    }

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    # Container for ensemble: model_name -> (oof_proba, test_proba, cv_score)
    results = {}

    for model_name, pipeline in models.items():
        print(f"\n=== Model: {model_name} ===")

        # Optional tuning for boosted models
        if tune and model_name in {'lgbm', 'xgb'}:
            print('Running RandomizedSearchCV tuning for', model_name)
            if model_name == 'lgbm':
                param_dist = {
                    'model__num_leaves': [31, 48, 64, 80],
                    'model__max_depth': [-1, 6, 8, 12],
                    'model__learning_rate': [0.01, 0.03, 0.05, 0.08],
                    'model__n_estimators': [200, 400, 600],
                    'model__subsample': [0.6, 0.8, 1.0],
                    'model__colsample_bytree': [0.6, 0.8, 1.0],
                }
            else:  # xgb
                param_dist = {
                    'model__max_depth': [4, 6, 8],
                    'model__learning_rate': [0.01, 0.03, 0.05],
                    'model__n_estimators': [200, 400, 600],
                    'model__subsample': [0.6, 0.8, 1.0],
                    'model__colsample_bytree': [0.6, 0.8, 1.0],
                }

            rnd = RandomizedSearchCV(
                pipeline,
                param_distributions=param_dist,
                n_iter=min(n_iter, 24),
                scoring='accuracy',
                cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=1,
            )
            rnd.fit(X, y)
            print('Best params:', rnd.best_params_)
            pipeline = rnd.best_estimator_

        n_classes = y.nunique()
        oof_proba = np.zeros((len(X), n_classes))
        test_proba = np.zeros((len(X_test), n_classes))

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f' Fold {fold+1}/{n_folds}')
            X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            model_fold = clone(pipeline)
            model_fold.fit(X_tr, y_tr)

            if hasattr(model_fold, 'predict_proba'):
                oof_proba[val_idx] = model_fold.predict_proba(X_val)
                test_proba += model_fold.predict_proba(X_test) / n_folds
            else:
                # fallback to predict and one-hot encode (less ideal)
                preds = model_fold.predict(X_val)
                oof_proba[val_idx] = np.eye(n_classes)[preds]
                test_preds = model_fold.predict(X_test)
                test_proba += np.eye(n_classes)[test_preds] / n_folds

            acc = accuracy_score(y_val, np.argmax(oof_proba[val_idx], axis=1))
            print(f'  Fold {fold+1} interim accuracy: {acc:.5f}')

        # Compute overall CV metrics from OOF
        oof_preds = np.argmax(oof_proba, axis=1)
        accuracy = accuracy_score(y, oof_preds)
        macro_f1 = f1_score(y, oof_preds, average='macro')
        print(f"CV accuracy (oof): {accuracy:.5f}, CV macro F1: {macro_f1:.5f}")

        # Refit final model on full train (pipeline may already be tuned)
        final_model = clone(pipeline)
        final_model.fit(X, y)
        if hasattr(final_model, 'predict_proba'):
            final_test_proba = final_model.predict_proba(X_test)
        else:
            final_test_preds = final_model.predict(X_test)
            final_test_proba = np.eye(n_classes)[final_test_preds]

        results[model_name] = {
            'oof_proba': oof_proba,
            'test_proba_cv_avg': test_proba,
            'test_proba_refit': final_test_proba,
            'accuracy': accuracy,
            'macro_f1': macro_f1,
        }

        # Persist OOF and test probas to disk for stacking / analysis
        npz_path = OUTPUT_DIR / f'{model_name}_probas.npz'
        np.savez_compressed(npz_path, oof=oof_proba, test_cv=test_proba, test_refit=final_test_proba)
        print('Saved probas to:', npz_path.name)

        # Save a per-model submission using the CV-averaged probabilities (soft voting)
        ensemble_test_proba = test_proba  # using CV average across folds
        ensemble_preds = np.argmax(ensemble_test_proba, axis=1)
        submission = sample_submission.copy()
        submission[TARGET] = ensemble_preds
        submission_path = OUTPUT_DIR / f'submission_{model_name}_cvavg.csv'
        submission.to_csv(submission_path, index=False)
        print('Wrote submission:', submission_path.name)

    # --- Simple ensemble across all model CV-averaged probabilities ---
    print('\nBuilding simple average ensemble across models (weighted by CV accuracy)')
    model_names = list(results.keys())
    n_models = len(model_names)
    n_classes = y.nunique()
    ensembled_proba = np.zeros((len(X_test), n_classes))

    # weight by accuracy to favor better models
    accs = np.array([results[m]['accuracy'] for m in model_names])
    weights = accs / accs.sum()

    for w, m in zip(weights, model_names):
        ensembled_proba += results[m]['test_proba_cv_avg'] * w

    ensembled_preds = np.argmax(ensembled_proba, axis=1)
    ensemble_sub = sample_submission.copy()
    ensemble_sub[TARGET] = ensembled_preds
    ensemble_path = OUTPUT_DIR / 'submission_ensemble_weighted.csv'
    ensemble_sub.to_csv(ensemble_path, index=False)
    print('Wrote ensemble submission:', ensemble_path.name)

    # Save CV summary
    rows = []
    for m in model_names:
        rows.append({'model': m, 'accuracy': results[m]['accuracy'], 'macro_f1': results[m]['macro_f1']})
    scores_df = pd.DataFrame(rows).sort_values(['accuracy', 'macro_f1'], ascending=False)
    scores_path = OUTPUT_DIR / 'cv_scores.csv'
    scores_df.to_csv(scores_path, index=False)
    print('Wrote CV scores to:', scores_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tune', type=lambda x: x.lower() in ('true', '1', 'yes'), default=False, help='Whether to run RandomizedSearchCV tuning for boosted models')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--n_iter', type=int, default=12, help='Number of RandomizedSearchCV iterations (when tuning)')
    args = parser.parse_args()

    main(tune=args.tune, n_folds=args.n_folds, n_iter=args.n_iter)
