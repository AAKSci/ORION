
# ORION V4 -- ALZHEIMERS DISEASE RISK MODEL
# Author      : Amir Kelson
# Version     : 4.0
# Last Updated: April 2026
# Purpose     : Train, validate, and evaluate an ML model
#               to predict Alzheimer's Disease diagnosis
#               from clinical and demographic features
#
# TRACEABILITY KEY

# STEP  = pipeline step number for reproducibility
# FIX   = documented bug fix with reason
# WHY   = explains a decision that might not be obvious



#  STEP 0: IMPORT LIBRARIES

import pandas as pd                          # W3REF: pandas -- for loading and manipulating tabular data (like Excel rows/columns)
import numpy as np                           # W3REF: numpy  -- for fast math operations on arrays and handling missing values (NaN)
from sklearn import linear_model             # sklearn = scikit-learn, the standard Python machine learning library
from sklearn.ensemble import RandomForestClassifier   # Random Forest: builds many decision trees and combines their votes
from sklearn.model_selection import (
    train_test_split,                        # Splits data into training set and test set
    StratifiedKFold,                         # K-Fold that keeps class proportions equal in each fold
    cross_val_score,                         # Runs a model across all k folds and returns AUC scores
    RandomizedSearchCV                       # Tries random combinations of hyperparameters to find the best ones
)
from sklearn.metrics import (
    confusion_matrix,                        # Table of true positives, false positives, etc.
    classification_report,                   # Prints precision, recall, F1 for each class
    roc_curve,                               # Calculates the ROC curve points (false positive rate vs true positive rate)
    auc                                      # Calculates the Area Under the Curve (AUC) -- higher = better model
)
import matplotlib.pyplot as plt              # W3REF: matplotlib -- Python's main chart/graph library
import seaborn as sns                        # Seaborn: builds on matplotlib to make prettier charts with less code
# ORION-FIXES-APPLIED
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier as _SkRF
from sklearn.linear_model import LogisticRegression as _SkLR

# FIX 3: models learn fill-in values (medians) from the data they are TRAINED on,
#        so test rows never influence imputation.
class _TrainOnlyImpute:
    def fit(self, X, y, sample_weight=None):
        X = pd.DataFrame(X)
        self.fill_ = X.median().fillna(0)
        return super().fit(X.mask(X.isna(), self.fill_, axis=1), y, sample_weight=sample_weight)
    def impute(self, X):
        return pd.DataFrame(X).pipe(lambda d: d.mask(d.isna(), self.fill_, axis=1))
    def predict(self, X):
        return super().predict(self.impute(X))
    def predict_proba(self, X):
        return super().predict_proba(self.impute(X))

class RandomForestClassifier(_TrainOnlyImpute, _SkRF): pass
class LogisticRegression(_TrainOnlyImpute, _SkLR):
    def decision_function(self, X):
        return super().decision_function(self.impute(X))


# STEP 1: DEFINE FILE PATHS 
# WHY: Storing paths as variables means we only change them in one place if files move.

original_path = '/Users/amirkelson/Downloads/'                          # Where main ADNI data lives
new_path      = '/Users/amirkelson/Downloads/Add.Feat.clean.Ad.Dtaswp/' # Where supplemental feature files live


#  STEP 2: LOAD DATA
# pd.read_csv() reads a CSV file into a DataFrame.
# A DataFrame is like a spreadsheet: rows = patients, columns = variables.

df      = pd.read_csv(original_path + 'all_subject_AD_ORION1_My_Table_02Mar2026.csv')  # Main ADNI dataset
moca_df = pd.read_csv(new_path + 'All_Subjects_MOCA_22Mar2026.csv')                    # MoCA cognitive scores
faq_df  = pd.read_csv(new_path + 'All_Subjects_FAQ_22Mar2026.csv')                     # FAQ functional scores (IADL)
dx_df   = pd.read_csv(new_path + 'All_Subjects_DXSUM_22Mar2026.csv')                   # Diagnosis summary (Parkinson, other dementias)

print("STEP 2 -- Data loaded")
print("Total rows loaded:", len(df))                         # len() counts rows -- W3REF: https://www.w3schools.com/python/ref_func_len.asp
print("MMSCORE non-null:", df['MMSCORE'].notna().sum())      # .notna() returns True where value exists, .sum() counts Trues


#  STEP 3: BUILD LONGITUDINAL COGNITIVE SLOPE
# WHY: Cognitive slope (MMSE change over time) is a strong predictor.
#      We calculate it by subtracting baseline MMSE from 12-month and 24-month MMSE.

df['visit'] = df['visit'].fillna('bl')           # fillna() replaces missing values -- treat missing visit as baseline ('bl')
df = df.sort_values(['subject_id', 'visit'])     # sort_values() orders rows -- W3REF: https://www.w3schools.com/python/pandas/ref_df_sort_values.asp

# Filter rows where visit == 'm12', keep only subject_id and MMSCORE columns
df_12 = df[df['visit'] == 'm12'][['subject_id', 'MMSCORE']].copy()  # .copy() prevents accidental edits to original df
df_12 = df_12.rename(columns={'MMSCORE': 'MMSCORE_12m'})            # rename() changes column name for clarity

df_24 = df[df['visit'] == 'm24'][['subject_id', 'MMSCORE']].copy()
df_24 = df_24.rename(columns={'MMSCORE': 'MMSCORE_24m'})

# groupby().first() keeps only the first visit row per patient
# FIX 2: take the BASELINE row only. groupby().first() took the first non-missing value
#        in each column, which could fill a baseline value from a later visit.
#        Only variables that never change (age at entry, sex, genotype, education) may come from another visit.
_static = df.groupby('subject_id')[['entry_age', 'PTGENDER', 'GENOTYPE', 'PTEDUCAT']].first()
_bl = df[df['visit'] == 'bl'].groupby('subject_id').first()
_sc = df[df['visit'] == 'sc'].groupby('subject_id').first()
df_first = _bl.combine_first(_sc.reindex(_bl.index))   # FIX: baseline = bl row, gaps filled from the screening (sc) visit
for _c in _static.columns:
    df_first[_c] = df_first[_c].fillna(_static[_c])
df_first = df_first.reset_index()
print("STEP 3 -- Rows after keeping first visit per patient:", len(df_first))

# merge() is like SQL JOIN -- links two tables on a shared column
df_first = df_first.merge(df_12, on='subject_id', how='left')  # how='left' keeps all rows from df_first even if no match
df_first = df_first.merge(df_24, on='subject_id', how='left')

# Calculate slope: how much did MMSE change from baseline?
# WHY: A negative slope means cognitive decline -- a key AD signal
df_first['cognitive_slope']    = df_first['MMSCORE_12m'] - df_first['MMSCORE']   # 12-month change
df_first['cognitive_slope_24'] = df_first['MMSCORE_24m'] - df_first['MMSCORE']   # 24-month change

print("cognitive_slope 12m non-null:", df_first['cognitive_slope'].notna().sum())
print("cognitive_slope 24m non-null:", df_first['cognitive_slope_24'].notna().sum())


# STEP 4: MERGE SUPPLEMENTAL FEATURES
# Filter MoCA and FAQ to baseline visit only, then merge into main dataset

moca_bl = moca_df[moca_df['VISCODE'] == 'bl'][['PTID', 'MOCA']].drop_duplicates('PTID').copy()
moca_bl = moca_bl.rename(columns={'MOCA': 'moca_score'})

faq_bl  = faq_df[faq_df['VISCODE'] == 'bl'][['PTID', 'FAQTOTAL']].drop_duplicates('PTID').copy()
faq_bl  = faq_bl.rename(columns={'FAQTOTAL': 'iadl_score'})   # IADL = Instrumental Activities of Daily Living

df_first = df_first.merge(moca_bl, left_on='subject_id', right_on='PTID', how='left')              # left_on/right_on used when columns have different names
df_first = df_first.merge(faq_bl,  left_on='subject_id', right_on='PTID', how='left', suffixes=('', '_faq'))  # suffixes prevent duplicate column name conflicts

dx_other = dx_df[dx_df['VISCODE'] == 'bl'][['PTID', 'DXPARK', 'DXOTHDEM']].drop_duplicates('PTID').copy()  # Parkinson's and other dementia flags
df_first = df_first.merge(dx_other, left_on='subject_id', right_on='PTID', how='left', suffixes=('', '_dx'))

print("STEP 4 -- Supplemental features merged")
print("moca_score non-null:", df_first['moca_score'].notna().sum())
print("iadl_score non-null:", df_first['iadl_score'].notna().sum())


# STEP 5: DROP ROWS MISSING ESSENTIAL VARIABLES

# WHY: We cannot impute age, gender, or APOE genotype -- these are non-negotiable inputs.
#      Any patient missing these is unusable and must be excluded.

hard_required = ['entry_age', 'PTGENDER', 'GENOTYPE']          # List of columns that must have real values
df_clean = df_first.dropna(subset=hard_required).copy()         # dropna(subset=) drops rows where any of these are missing
print("STEP 5 -- Rows after hard drop:", len(df_clean))


#  STEP 6: FIX 1 -- EXCLUDE OTHER DEMENTIAS
# FIX: Parkinson's disease dementia and other non-AD dementias were in the "Not AD" group,
#      making the boundary between AD and controls artificially noisy.
# WHY: If a Parkinson's patient looks like an AD patient on our features,
#      the model gets confused. Excluding them sharpens the classification task.

before_exclusion = len(df_clean)
df_clean = df_clean[df_clean['DXPARK'].isna()   | (df_clean['DXPARK']   != 1)]  # Keep rows where DXPARK is missing OR not equal to 1
df_clean = df_clean[df_clean['DXOTHDEM'].isna() | (df_clean['DXOTHDEM'] != 1)]  # Same for other dementias
print("STEP 6 -- Rows after excluding other dementias:", len(df_clean))
print("Patients excluded:", before_exclusion - len(df_clean))


#  STEP 7: IMPUTE MISSING VALUES 
# Imputation = filling in missing values with a reasonable estimate.
# WHY: Most ML models cannot handle NaN (missing) values -- they will crash or produce wrong results.
#      median() is used (not mean) because it is not affected by extreme outliers.

df_clean['cognitive_slope']    = df_clean['cognitive_slope'].fillna(0)                           # WHY: If no 12m visit, assume no change (slope = 0)
df_clean['cognitive_slope_24'] = df_clean['cognitive_slope_24'].fillna(0)                        # Same for 24-month slope

# mode() returns the most common value -- used for categorical columns


#  STEP 8: FIX 2 -- MOCA REAL vs IMPUTED FLAG
# FIX: MoCA was missing for many patients. Instead of just imputing,
#      we add a binary flag column so the model knows which scores are real.
# WHY: Imputed scores (median) carry no real information.
#      The flag lets the model discount imputed values appropriately.

df_clean['moca_real']  = df_clean['moca_score'].notna().astype(int)          # 1 if patient has a real MoCA score, 0 if it will be imputed

print("STEP 8 -- MoCA flag created")
print("Patients with real MoCA scores:", df_clean['moca_real'].sum())
print("Patients with imputed MoCA:    ", (df_clean['moca_real'] == 0).sum())


#STEP 9: FEATURE ENGINEERING
# Feature engineering = creating new columns from existing ones to capture relationships.

df_clean = df_clean.rename(columns={'MMSCORE': 'mmse_score'})   # rename for clarity throughout the rest of the script

# Convert height/weight to metric and calculate BMI
# BMI = weight_kg / height_m^2  (standard WHO formula)
metric = df_clean['VSHEIGHT'] > 100
df_clean['height_m']  = np.where(metric, df_clean['VSHEIGHT'] / 100, df_clean['VSHEIGHT'] * 0.0254)
df_clean['weight_kg'] = np.where(metric, df_clean['VSWEIGHT'], df_clean['VSWEIGHT'] * 0.453592)
df_clean['BMI'] = df_clean['weight_kg'] / (df_clean['height_m'] ** 2)
df_clean.loc[(df_clean['BMI'] < 15) | (df_clean['BMI'] > 50), 'BMI'] = np.nan
print("BMI plausible values:", df_clean['BMI'].notna().sum(), "| set to missing:", df_clean['BMI'].isna().sum())
print("BMI after fix: mean", round(df_clean['BMI'].mean(), 1), "SD", round(df_clean['BMI'].std(), 1))

# APOE4: does the patient's genotype contain a '4' allele?
# WHY: APOE4 is the strongest known genetic risk factor for late-onset AD
df_clean['APOE4']  = df_clean['GENOTYPE'].str.contains('4').astype(int)   # str.contains() searches string -- returns True/False, astype(int) converts to 1/0
df_clean['gender'] = df_clean['PTGENDER'].map({1.0: 0, 2.0: 1}).fillna(0) # map() replaces values -- 1=male->0, 2=female->1

# get_dummies() converts a categorical column into binary 0/1 columns (one-hot encoding)
hypertension = pd.DataFrame({'hypertension_1': (df_clean['MH4CARD'] == 1).astype(float).where(df_clean['MH4CARD'].notna())}, index=df_clean.index)  # FIX 3: missing stays missing  # drop_first=True avoids dummy variable trap

# Rename remaining columns for human readability
df_clean = df_clean.rename(columns={
    'entry_age' : 'age',
    'PTEDUCAT'  : 'education_years',
    'DXNORM'    : 'diagnosis_normal',
    'DXMCI'     : 'Mild_Cognitive_Impairment',
    'DXAD'      : 'Alzheimers_Diagnosis'
})

# Create binary outcome: 1 = Alzheimer's, 0 = Normal or MCI, NaN = unknown
# np.where() is like IF-THEN-ELSE
df_clean['AD_binary'] = np.where(
    df_clean['Alzheimers_Diagnosis'] == 1.0, 1,           # IF diagnosed AD -> 1
    np.where(df_clean['diagnosis_normal'] == 1.0, 0,      # ELSE IF normal -> 0
    np.where(df_clean['Mild_Cognitive_Impairment'] == 1.0, 0,  # ELSE IF MCI -> 0 (not AD)
    np.nan)))                                              # ELSE -> unknown, exclude from model


#  STEP 10: FIX 3 -- INTERACTION TERMS (APOE4 MULTICOLLINEARITY FIX) 
# FIX: Original code included APOE4_x_gender which caused APOE4 coefficient
#      to appear protective (negative) when it should increase risk (positive).
#      This is multicollinearity -- two features that are too correlated confuse the model.
# WHY: APOE4_x_age is clinically validated (risk increases with age for APOE4 carriers).
#      APOE4_x_gender was removed because it introduced more noise than signal.

df_clean['APOE4_x_age']       = df_clean['APOE4']           * df_clean['age']             # APOE4 risk amplified by age
df_clean['age_x_gender']      = df_clean['age']             * df_clean['gender']           # Age effect differs by sex
df_clean['mmse_x_age']        = df_clean['mmse_score']      * df_clean['age']             # MMSE score in context of patient age
df_clean['iadl_x_gender']     = df_clean['iadl_score']      * df_clean['gender']           # Functional decline differs by sex
df_clean['education_x_APOE4'] = df_clean['education_years'] * df_clean['APOE4']           # Cognitive reserve modifies APOE4 risk

print("STEP 10 -- Interaction terms created (APOE4_x_gender excluded to fix multicollinearity)")


# ── STEP 11: SAVE STRATIFIED SUBSETS 
# WHY: We will train separate models for male and female patients
#      because AD risk factors (especially hormonal) differ by sex.

df_male   = df_clean[df_clean['gender'] == 0].copy()   # Boolean indexing: keep rows where gender == 0 (male)
df_female = df_clean[df_clean['gender'] == 1].copy()   # Keep rows where gender == 1 (female)
df_male.to_csv(original_path   + 'Orion_adni_MALE.csv',   index=False)   # Save to CSV, index=False omits row numbers
df_female.to_csv(original_path + 'Orion_adni_FEMALE.csv', index=False)
print("STEP 11 -- Male patients:  ", len(df_male))
print("STEP 11 -- Female patients:", len(df_female))


# ── STEP 12: DEFINE FEATURE LISTS

# Lists store multiple items in a single variable.
# WHY: Two feature lists support the primary model (full features)
#      and a fallback model (no MMSE required) for missing-data scenarios.

primary_features = [
    'age', 'gender', 'education_years', 'BMI', 'APOE4',
    'mmse_score', 'moca_score', 'moca_real', 'iadl_score',
    'cognitive_slope', 'cognitive_slope_24',
    'APOE4_x_age', 'age_x_gender',
    'mmse_x_age', 'iadl_x_gender', 'education_x_APOE4'
]

fallback_features = [                                  # No mmse_score, mmse_x_age, or cognitive_slope_24
    'age', 'gender', 'education_years', 'BMI', 'APOE4',
    'moca_score', 'moca_real', 'iadl_score',
    'APOE4_x_age',                                  # FIX: cognitive_slope removed (it is MMSE-derived)
    'age_x_gender', 'iadl_x_gender', 'education_x_APOE4'
]


# ── STEP 13: BUILD FINAL FEATURE MATRIX 
# pd.concat() stacks DataFrames side by side (axis=1) or top to bottom

X = pd.concat([df_clean[primary_features], hypertension], axis=1)  # X = feature matrix (inputs to model)
Y = df_clean['AD_binary']                                           # Y = target variable (what we are predicting)

mask = Y.notna()             # Boolean mask: True where Y has a real diagnosis label
X    = X[mask]               # Keep only patients with known diagnosis -- W3REF: boolean indexing
Y    = Y[mask]

print("STEP 13 -- Feature matrix built")
print("X shape:", X.shape)       # .shape returns (rows, columns) -- W3REF: https://www.w3schools.com/python/pandas/ref_df_shape.asp
print("Y counts:\n", Y.value_counts())  # value_counts() shows how many patients have each diagnosis label


#  STEP 14: HYPERPARAMETER TUNING 
# WHY: Random Forest has several settings (hyperparameters) that control how complex
#      the model is. Without tuning, the default settings often cause overfitting.
#      Overfitting = model memorizes training data but fails on new patients.
#
# KEY HYPERPARAMETERS:
#   n_estimators      = number of trees in the forest (more = slower but more stable)
#   max_depth         = how deep each tree can grow (deeper = more risk of memorization)
#   min_samples_split = minimum patients needed to split a branch (higher = simpler trees)
#   min_samples_leaf  = minimum patients required in each leaf node
#   max_features      = how many features each tree considers at each split
#
# RandomizedSearchCV tests random combinations from the param_grid and uses
# cross-validation to score each combination, returning the best one.

print("\nSTEP 14 -- Hyperparameter tuning (RandomizedSearchCV, 5-fold CV inside)")
print("This may take 1-2 minutes...")

param_grid = {
    'n_estimators'     : [100, 200, 300],
    'max_depth'        : [3, 5, 7, 10],   # None = unlimited depth -- WARNING: this can cause overfitting
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf' : [1, 2, 4],
    'max_features'     : ['sqrt', 'log2']
}

rf_search = RandomizedSearchCV(
    RandomForestClassifier(class_weight='balanced', random_state=42),
    param_distributions = param_grid,
    n_iter              = 30,
    cv                  = 5,
    scoring             = 'roc_auc',
    random_state        = 42,
    n_jobs = 1
)
# FIX 1: split FIRST, tune on the training set only (same split as Step 16)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
rf_search.fit(X_train, Y_train)

best_params = rf_search.best_params_
best_cv_auc = rf_search.best_score_

print("Best hyperparameters found:")
for k, v in best_params.items():
    print(f"  {k}: {v}")
print(f"Best cross-validated AUC during tuning: {round(best_cv_auc, 3)}")
print("NOTE: If best max_depth is None, try capping at 10 and compare AUC -- unlimited depth can overfit")


# STEP 15: K-FOLD CROSS VALIDATION 
# WHY: A single 80/20 train-test split can give misleading results if the split is lucky or unlucky.
#      K-Fold solves this by splitting data into k groups, training on k-1 groups, testing on 1 group,
#      and rotating through all combinations. You get k AUC scores instead of just 1.
#
# INTERPRETING RESULTS:
#   Mean AUC   = the model's true average performance
#   Std Dev    = how stable the model is across different data splits
#     Std Dev < 0.03         = stable model (good)
#     Std Dev 0.03 to 0.06   = moderate variance (acceptable)
#     Std Dev > 0.06         = unstable model (warning -- may be overfitting)
#
# StratifiedKFold ensures each fold has the same proportion of AD vs Not-AD patients.
# This is BETTER than plain KFold because it prevents a fold from being all AD or all Not-AD.


print("\nSTEP 15 -- 5-Fold Stratified Cross Validation")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

lr_cv_scores = cross_val_score(
    LogisticRegression(max_iter=5000, class_weight='balanced'),
    X, Y, cv=cv, scoring='roc_auc'
)
print("\nLogistic Regression -- 5-Fold AUC per fold:", [round(s, 3) for s in lr_cv_scores])
print(f"  Mean AUC : {round(lr_cv_scores.mean(), 3)}")
print(f"  Std Dev  : {round(lr_cv_scores.std(), 3)}")

# FIX 1: NESTED CV -- the hyperparameter search is redone inside every outer fold,
#        so each fold's score comes from data the tuning never saw. (Takes several minutes.)
print("Running nested CV for the tuned random forest...")
rf_tuned_cv_scores = cross_val_score(
    RandomizedSearchCV(RandomForestClassifier(class_weight='balanced', random_state=42),
                       param_distributions=param_grid, n_iter=30, cv=5,
                       scoring='roc_auc', random_state=42, n_jobs=1),
    X, Y, cv=cv, scoring='roc_auc'
)
print("\nRandom Forest (TUNED, NESTED CV) -- 5-Fold AUC per fold:", [round(s, 3) for s in rf_tuned_cv_scores])
print(f"  Mean AUC : {round(rf_tuned_cv_scores.mean(), 3)}")
print(f"  Std Dev  : {round(rf_tuned_cv_scores.std(), 3)}")

rf_untuned_cv_scores = cross_val_score(
    RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    X, Y, cv=cv, scoring='roc_auc'
)
print("\nRandom Forest (UNTUNED, no depth limit) -- 5-Fold AUC per fold:", [round(s, 3) for s in rf_untuned_cv_scores])
print(f"  Mean AUC : {round(rf_untuned_cv_scores.mean(), 3)}")
print(f"  Std Dev  : {round(rf_untuned_cv_scores.std(), 3)}")
print("  WHY: Compare tuned vs untuned -- if untuned is higher, unlimited depth was inflating results")


# ── STEP 16: TRAIN-TEST SPLIT AND OVERFITTING DIAGNOSTIC ────

# train_test_split() divides data into two groups:
#   Training set (80%) = data the model learns from
#   Test set (20%)     = data the model has never seen, used to evaluate real performance

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size    = 0.2,
    random_state = 42,
    stratify     = Y
)
print("\nSTEP 16 -- Train-Test Split")
print("Training rows:", len(X_train))
print("Testing rows: ", len(X_test))

# OVERFITTING DIAGNOSTIC
# WHY: The most reliable sign of overfitting is when training AUC >> test AUC.
#      Gap > 0.05 = overfitting warning.

print("\nSTEP 16b -- Overfitting Diagnostic (Training AUC vs Test AUC)")

logr = LogisticRegression(max_iter=5000, class_weight='balanced')
logr.fit(X_train, Y_train)
lr_train_auc = auc(*roc_curve(Y_train, logr.predict_proba(X_train)[:, 1])[:2])
lr_test_auc  = auc(*roc_curve(Y_test,  logr.predict_proba(X_test)[:, 1])[:2])
lr_gap       = lr_train_auc - lr_test_auc
print(f"\nLogistic Regression:")
print(f"  Training AUC : {round(lr_train_auc, 3)}")
print(f"  Test AUC     : {round(lr_test_auc, 3)}")
print(f"  Gap          : {round(lr_gap, 3)}  {'<-- WARNING: possible overfitting' if lr_gap > 0.05 else '<-- OK: gap is acceptable'}")

rf_tuned = RandomForestClassifier(**best_params, class_weight='balanced', random_state=42)
rf_tuned.fit(X_train, Y_train)
rf_train_auc = auc(*roc_curve(Y_train, rf_tuned.predict_proba(X_train)[:, 1])[:2])
rf_test_auc  = auc(*roc_curve(Y_test,  rf_tuned.predict_proba(X_test)[:, 1])[:2])
rf_gap       = rf_train_auc - rf_test_auc
print(f"\nRandom Forest (TUNED, max_depth={best_params.get('max_depth')}):")
print(f"  Training AUC : {round(rf_train_auc, 3)}")
print(f"  Test AUC     : {round(rf_test_auc, 3)}")
print(f"  Gap          : {round(rf_gap, 3)}  {'<-- WARNING: possible overfitting' if rf_gap > 0.05 else '<-- OK: gap is acceptable'}")

rf_untuned = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_untuned.fit(X_train, Y_train)
rfu_train_auc = auc(*roc_curve(Y_train, rf_untuned.predict_proba(X_train)[:, 1])[:2])
rfu_test_auc  = auc(*roc_curve(Y_test,  rf_untuned.predict_proba(X_test)[:, 1])[:2])
rfu_gap       = rfu_train_auc - rfu_test_auc
print(f"\nRandom Forest (UNTUNED, no depth limit -- original settings):")
print(f"  Training AUC : {round(rfu_train_auc, 3)}")
print(f"  Test AUC     : {round(rfu_test_auc, 3)}")
print(f"  Gap          : {round(rfu_gap, 3)}  {'<-- WARNING: possible overfitting' if rfu_gap > 0.05 else '<-- OK'}")
print("  WHY: A training AUC near 1.000 with unlimited depth is a hallmark of overfitting")

lr_predicted = logr.predict(X_test)
lr_proba     = logr.predict_proba(X_test)[:, 1]
rf_predicted = rf_tuned.predict(X_test)
rf_proba     = rf_tuned.predict_proba(X_test)[:, 1]


# ── STEP 17: FALLBACK MODEL (no MMSE required) 
# WHY: Some patients in real clinical settings will not have MMSE on file.
#      The fallback model uses only features available without MMSE.
#      This makes ORION deployable in data-limited settings.

df_fallback = df_clean.copy()
X_fb = pd.concat([
    df_fallback[fallback_features],
    hypertension.loc[df_fallback.index]
], axis=1)
Y_fb    = df_fallback['AD_binary']
mask_fb = Y_fb.notna()
X_fb    = X_fb[mask_fb]
Y_fb    = Y_fb[mask_fb]

if len(Y_fb.unique()) >= 2:
    X_fb_train, X_fb_test, Y_fb_train, Y_fb_test = train_test_split(
        X_fb, Y_fb, test_size=0.2, random_state=42, stratify=Y_fb
    )
    rf_fallback = RandomForestClassifier(**best_params, class_weight='balanced', random_state=42)
    rf_fallback.fit(X_fb_train, Y_fb_train)
    fb_predicted      = rf_fallback.predict(X_fb_test)
    fb_proba          = rf_fallback.predict_proba(X_fb_test)[:, 1]
    fpr_fb, tpr_fb, _ = roc_curve(Y_fb_test, fb_proba)
    auc_fb            = auc(fpr_fb, tpr_fb)
    print("\nSTEP 17 -- FALLBACK MODEL (no MMSE required)")
    print("Fallback Model AUC:", round(auc_fb, 3))
    print(classification_report(Y_fb_test, fb_predicted, target_names=['Not AD', 'Alzheimers']))


# ── STEP 18: STRATIFIED MODELS (male / female) 
# WHY: Sex-stratified models allow us to report gender-specific AUC,
#      which is important for equity in AI and for publication credibility.

def run_stratified_model(data, label):
    data = data.copy()
    X_s = pd.concat([
        data[primary_features],
        hypertension.loc[data.index]
    ], axis=1)
    Y_s    = data['AD_binary']
    mask_s = Y_s.notna()
    X_s    = X_s[mask_s]
    Y_s    = Y_s[mask_s]

    if len(Y_s.unique()) < 2:
        print(label, "skipped -- only one class present")
        return None, None, None, None

    X_tr, X_te, Y_tr, Y_te = train_test_split(
        X_s, Y_s, test_size=0.2, random_state=42, stratify=Y_s
    )
    model = RandomForestClassifier(**best_params, class_weight='balanced', random_state=42)
    model.fit(X_tr, Y_tr)

    tr_auc = auc(*roc_curve(Y_tr, model.predict_proba(X_tr)[:, 1])[:2])
    te_auc = auc(*roc_curve(Y_te, model.predict_proba(X_te)[:, 1])[:2])
    print(f"{label} -- Train AUC: {round(tr_auc,3)} | Test AUC: {round(te_auc,3)} | Gap: {round(tr_auc-te_auc,3)}  {'<-- WARNING' if (tr_auc-te_auc) > 0.05 else '<-- OK'}")

    preds           = model.predict(X_te)
    probs           = model.predict_proba(X_te)[:, 1]
    fpr_s, tpr_s, _ = roc_curve(Y_te, probs)
    auc_s           = auc(fpr_s, tpr_s)
    print(classification_report(Y_te, preds, target_names=['Not AD', 'Alzheimers']))
    return fpr_s, tpr_s, auc_s, model


print("\nSTEP 18 -- MALE MODEL")
fpr_m, tpr_m, auc_m, rf_male = run_stratified_model(df_male, "Male")

print("\nSTEP 18 -- FEMALE MODEL")
fpr_f, tpr_f, auc_f, rf_female = run_stratified_model(df_female, "Female")


# ── STEP 19: FINAL PERFORMANCE REPORTS 

print("\nSTEP 19 -- LOGISTIC REGRESSION REPORT (test set)")
print(classification_report(Y_test, lr_predicted, target_names=['Not AD', 'Alzheimers']))

fpr_lr, tpr_lr, _ = roc_curve(Y_test, lr_proba)
fpr_rf, tpr_rf, _ = roc_curve(Y_test, rf_proba)
auc_lr = auc(fpr_lr, tpr_lr)
auc_rf = auc(fpr_rf, tpr_rf)

print("Logistic Regression Test AUC:", round(auc_lr, 3))
print("\nSTEP 19 -- RANDOM FOREST REPORT (tuned, test set)")
print(classification_report(Y_test, rf_predicted, target_names=['Not AD', 'Alzheimers']))
print("Random Forest Test AUC:", round(auc_rf, 3))


# ── STEP 20: PILOT STUDY STATISTICS

total_patients  = len(Y)
ad_patients     = int(Y.sum())
not_ad_patients = int(len(Y) - Y.sum())
test_patients   = len(Y_test)

print("\nSTEP 20 -- PILOT STUDY STATISTICS")
print("Total patients with known diagnosis:", total_patients)
print("AD patients:                        ", ad_patients)
print("Not AD patients:                    ", not_ad_patients)
print("Test set size:                      ", test_patients)
print(f"LR  -- 5-Fold CV AUC: {round(lr_cv_scores.mean(),3)} +/- {round(lr_cv_scores.std(),3)}")
print(f"RF  -- 5-Fold CV AUC: {round(rf_tuned_cv_scores.mean(),3)} +/- {round(rf_tuned_cv_scores.std(),3)}")
print("Recommended validation set needed:  500+ patients (NACC or OASIS-3)")
print("Recommended next step:              External validation on NACC / OASIS-3")

df_clean.to_csv(original_path + 'Orion_adni_clean_v2.csv', index=False)
print("\nSTEP 20 -- Done.", len(df_clean), "records saved to Orion_adni_clean_v2.csv")



# STEP 21 — ORION V4 FULL VISUALIZATION SUITE
# Saves ALL charts individually to Downloads


import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from mpl_toolkits.mplot3d import Axes3D
import subprocess
import sys

# ── ENSURE SHAP IS INSTALLED
try:
    import shap
except ImportError:
    print("Installing SHAP...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "shap"])
    import shap

# ── DEFINE SAVE PATH 
original_path = "/Users/amirkelson/Downloads/"
os.makedirs(original_path, exist_ok=True)

print("\nAll charts will be saved to:")
print(original_path)

sns.set_style("whitegrid")


#  FEATURE COEFFICIENTS — LOGISTIC REGRESSION


plt.figure(figsize=(10,8))

features_list = X.columns.tolist()
coefs = logr.coef_[0]

colors = []
for c in coefs:
    if c > 0:
        colors.append("red")
    else:
        colors.append("blue")

plt.barh(features_list, coefs, color=colors)
plt.axvline(x=0)
plt.title("Feature Coefficients — Logistic Regression")
plt.xlabel("Coefficient Value")

coef_path = os.path.join(original_path, "ORION_Feature_Coefficients.png")
plt.savefig(coef_path, dpi=300, bbox_inches="tight")
print("Saved:", coef_path)
plt.show()


#  FEATURE IMPORTANCE — RANDOM FOREST


plt.figure(figsize=(10,8))

importances = rf_tuned.feature_importances_

importance_pairs = list(zip(features_list, importances))
importance_pairs.sort(key=lambda x: x[1])

sorted_features = []
sorted_values = []

for pair in importance_pairs:
    sorted_features.append(pair[0])
    sorted_values.append(pair[1])

plt.barh(sorted_features, sorted_values)
plt.title("Feature Importance — Random Forest (Tuned)")
plt.xlabel("Importance Score")

fi_path = os.path.join(original_path, "ORION_RF_Feature_Importance.png")
plt.savefig(fi_path, dpi=300, bbox_inches="tight")
print("Saved:", fi_path)
plt.show()


#  OUTCOME DISTRIBUTION


plt.figure(figsize=(6,6))

counts = Y.value_counts()

plt.bar(["Not AD (0)", "AD (1)"], counts)
plt.title("Outcome Distribution")
plt.ylabel("Patient Count")

outcome_path = os.path.join(original_path, "ORION_Outcome_Distribution.png")
plt.savefig(outcome_path, dpi=300, bbox_inches="tight")
print("Saved:", outcome_path)
plt.show()


# 4 ROC CURVE


plt.figure(figsize=(8,6))

plt.plot(fpr_lr, tpr_lr, label="Logistic Regression")
plt.plot(fpr_rf, tpr_rf, label="Random Forest")
plt.plot([0,1], [0,1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Model Comparison")
plt.legend()

roc_path = os.path.join(original_path, "ORION_ROC_Curve.png")
plt.savefig(roc_path, dpi=300, bbox_inches="tight")
print("Saved:", roc_path)
plt.show()


#  CONFUSION MATRIX — LOGISTIC REGRESSION


cm_lr = confusion_matrix(Y_test, lr_predicted)

plt.figure(figsize=(6,5))
sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix — Logistic Regression")

cm_lr_path = os.path.join(original_path, "ORION_CM_Logistic.png")
plt.savefig(cm_lr_path, dpi=300, bbox_inches="tight")
print("Saved:", cm_lr_path)
plt.show()


#  CONFUSION MATRIX — RANDOM FOREST


cm_rf = confusion_matrix(Y_test, rf_predicted)

plt.figure(figsize=(6,5))
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Greens")
plt.title("Confusion Matrix — Random Forest")

cm_rf_path = os.path.join(original_path, "ORION_CM_RandomForest.png")
plt.savefig(cm_rf_path, dpi=300, bbox_inches="tight")
print("Saved:", cm_rf_path)
plt.show()


#  3D GRADIENT DESCENT VISUALIZATION


# Create synthetic loss surface for visualization
w1 = np.linspace(-3, 3, 100)
w2 = np.linspace(-3, 3, 100)

W1, W2 = np.meshgrid(w1, w2)

Loss = W1**2 + W2**2  # simple convex surface

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection="3d")

ax.plot_surface(W1, W2, Loss, cmap="viridis")

ax.set_xlabel("Weight 1")
ax.set_ylabel("Weight 2")
ax.set_zlabel("Loss")
ax.set_title("3D Gradient Descent Loss Surface")

gd_path = os.path.join(original_path, "ORION_3D_Gradient_Descent.png")
plt.savefig(gd_path, dpi=300, bbox_inches="tight")
print("Saved:", gd_path)
plt.show()


# 
# # SHAP VALUES — RANDOM FOREST (FIXED + ROBUST)

import shap
print("\nGenerating SHAP values...")

explainer = shap.TreeExplainer(rf_tuned)
shap_values = explainer(rf_tuned.impute(X_test))
plt.close("all")
plt.figure()
shap.plots.beeswarm(shap_values[:, :, 1], max_display=17, show=False)
shap_path = os.path.join(original_path, "ORION_SHAP_Summary.png")
plt.savefig(shap_path, dpi=300, bbox_inches="tight")
plt.close("all")
print("Saved:", shap_path)
mean_abs = np.abs(shap_values.values[:, :, 1]).mean(axis=0)
print("\nMEAN |SHAP| ON TEST SET (highest first):")
for f, v in sorted(zip(X_test.columns, mean_abs), key=lambda t: -t[1]):
    print(f"  {f}: {v:.4f}")
print("\nRANDOM FOREST IMPORTANCE (exact):")
for f, v in sorted(zip(X.columns, rf_tuned.feature_importances_), key=lambda t: -t[1]):
    print(f"  {f}: {v:.4f}")
print("\nLOGISTIC REGRESSION COEFFICIENTS:")
for f, v in sorted(zip(X.columns, logr.coef_[0]), key=lambda t: -abs(t[1])):
    print(f"  {f}: {v:.4f}")


# STEP 21 COMPLETE


print("\nSTEP 21 COMPLETE — ALL VISUALIZATIONS GENERATED")
print("Check your Downloads folder.")
# ============================================================
# STEP 22 — MODEL PACKAGING (artifacts + interface + metadata)
# ============================================================
import pickle, yaml
from pathlib import Path

PACKAGE_DIR = Path(original_path) / "ORION_v4_package"
PACKAGE_DIR.mkdir(exist_ok=True)

# ── 22a: SAVE MODEL ARTIFACTS 
with open(PACKAGE_DIR / "rf_tuned.pkl",   "wb") as f: pickle.dump(rf_tuned,   f)
with open(PACKAGE_DIR / "rf_fallback.pkl","wb") as f: pickle.dump(rf_fallback, f)
with open(PACKAGE_DIR / "logr.pkl",       "wb") as f: pickle.dump(logr,        f)
with open(PACKAGE_DIR / "feature_columns.pkl", "wb") as f: pickle.dump(X.columns.tolist(), f)
print("STEP 22a -- Model artifacts saved")

# ── 22b: REQUIREMENTS FIL
(PACKAGE_DIR / "requirements.txt").write_text(
    "scikit-learn==1.4.2\npandas==2.2.2\nnumpy==1.26.4\nshap==0.45.0\nmatplotlib==3.8.4\npyyaml==6.0.1\n"
)

# ── 22c: MODEL INTERFACE FILE 
interface_code = '''
import pickle
from pathlib import Path
import pandas as pd, numpy as np

PACKAGE_PATH = Path(__file__).parent

class OrionModel:
    def __init__(self, model_file="rf_tuned.pkl"):
        with open(PACKAGE_PATH / model_file, "rb") as f:
            self.model = pickle.load(f)
        with open(PACKAGE_PATH / "feature_columns.pkl", "rb") as f:
            self.features = pickle.load(f)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(df[self.features].fillna(df[self.features].median()))

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Returns AD risk probability (column 1 = AD risk score 0-1)."""
        return self.model.predict_proba(df[self.features].fillna(df[self.features].median()))

def load_model(model_file="rf_tuned.pkl"):
    return OrionModel(model_file)
'''
(PACKAGE_DIR / "interface.py").write_text(interface_code)

# ── 22d: METADATA YAML
metadata = {
    "name"        : "ORION v4 — Alzheimer's Disease Risk Model",
    "version"     : "4.0",
    "author"      : "Amir Kelson",
    "created_date": "April 2026",
    "dataset"     : "ADNI (all_subject_AD_ORION1_My_Table_02Mar2026.csv)",
    "primary_model": "RandomForestClassifier (tuned)",
    "fallback_model": "RandomForestClassifier (no MMSE required)",
    "hyperparameters": best_params,
    "cv_auc_mean" : round(rf_tuned_cv_scores.mean(), 3),
    "cv_auc_std"  : round(rf_tuned_cv_scores.std(),  3),
    "input_features": primary_features,
    "output": {
        "name"       : "AD_risk_probability",
        "type"       : "float",
        "description": "Predicted probability of Alzheimer's diagnosis (0=low risk, 1=high risk)"
    },
    "exclusions": ["DXPARK==1 (Parkinson's)", "DXOTHDEM==1 (other dementias)"],
    "notes": "Sex-stratified models also packaged. Use fallback model when MMSE unavailable."
}
with open(PACKAGE_DIR / "metadata.yaml", "w") as f:
    yaml.dump(metadata, f, default_flow_style=False)

print("STEP 22 -- Package written to:", PACKAGE_DIR)
print("  rf_tuned.pkl | rf_fallback.pkl | logr.pkl | feature_columns.pkl")
print("  interface.py | requirements.txt | metadata.yaml")



# STEP 23 — GROUP-LEVEL RISK PROBABILITY REPORT
# Every participant gets an AD risk score + risk tier


risk_df = X.copy()
risk_df['subject_id']       = df_clean.loc[mask, 'subject_id'].values
risk_df['true_label']       = Y.values
# FIX 4: each participant is scored by a model that never trained on them (out-of-fold)
risk_df['AD_risk_prob']     = cross_val_predict(
    RandomForestClassifier(**best_params, class_weight='balanced', random_state=42),
    X, Y, cv=cv, method='predict_proba')[:, 1]
risk_df['predicted_AD']     = (risk_df['AD_risk_prob'] >= 0.5).astype(int)

# Risk tiers: Low / Moderate / High
risk_df['risk_tier'] = pd.cut(
    risk_df['AD_risk_prob'],
    bins   = [0, 0.33, 0.66, 1.0],
    labels = ["Low", "Moderate", "High"],
    include_lowest = True                                       # FIX: a probability of exactly 0 was dropped
)

# Sort highest risk first
risk_df = risk_df.sort_values('AD_risk_prob', ascending=False).reset_index(drop=True)

# Save full table
risk_csv = original_path + "ORION_participant_risk_scores.csv"
risk_df[['subject_id','AD_risk_prob','risk_tier','predicted_AD','true_label']].to_csv(risk_csv, index=False)
print("\nSTEP 23 -- Risk scores saved:", risk_csv)

# Summary by tier
print("\nRisk Tier Summary:")
print(risk_df.groupby('risk_tier', observed=True)['subject_id'].count().rename("n_patients"))

# Bar chart — distribution of risk scores
plt.figure(figsize=(9,5))
plt.hist(risk_df['AD_risk_prob'], bins=30, color='steelblue', edgecolor='white')
plt.axvline(0.33, color='orange', linestyle='--', label='Low/Moderate cutoff')
plt.axvline(0.66, color='red',    linestyle='--', label='Moderate/High cutoff')
plt.xlabel("AD Risk Probability")
plt.ylabel("Number of Participants")
plt.title("ORION v4 — Group Risk Probability Distribution")
plt.legend()
dist_path = original_path + "ORION_Risk_Distribution.png"
plt.savefig(dist_path, dpi=300, bbox_inches='tight')
plt.show()
print("Saved:", dist_path)



# STEP 24 — PER-PARTICIPANT SHAP WATERFALL PLOTS
# Shows each individual's contributing risk factors
# Top 5 highest-risk + top 5 lowest-risk participants


explainer_pkg  = shap.TreeExplainer(rf_tuned)
shap_vals_full = explainer_pkg(rf_tuned.impute(X))   # SHAP Explanation object (new API)

# Helper: waterfall for one participant
def plot_participant_waterfall(idx, subject_id, risk_prob, risk_tier, save_path):
    plt.figure()
    shap.plots.waterfall(shap_vals_full[idx, :, 1], max_display=12, show=False)
    plt.title(
        f"Participant: {subject_id}\n"
        f"AD Risk: {risk_prob:.1%}  |  Tier: {risk_tier}",
        fontsize=11
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved waterfall: {save_path}")

waterfall_dir = Path(original_path) / "ORION_Waterfall_Plots"
waterfall_dir.mkdir(exist_ok=True)

# Map subject_id back to X index position
subj_to_xidx = {row['subject_id']: i for i, row in risk_df.iterrows()}

# Top 5 highest risk
print("\nSTEP 24 -- Generating waterfall plots (top 5 high risk + top 5 low risk)...")
for rank, row in risk_df.head(5).iterrows():
    x_idx    = list(X.index).index(
        df_clean.loc[mask].index[df_clean.loc[mask,'subject_id'] == row['subject_id']].tolist()[0]
    ) if row['subject_id'] in df_clean.loc[mask,'subject_id'].values else None
    if x_idx is None: continue
    fname = waterfall_dir / f"HIGH_risk_{rank+1}_{row['subject_id']}.png"
    plot_participant_waterfall(x_idx, row['subject_id'], row['AD_risk_prob'], row['risk_tier'], fname)

# Top 5 lowest risk
for rank, row in risk_df.tail(5).iterrows():
    x_idx = list(X.index).index(
        df_clean.loc[mask].index[df_clean.loc[mask,'subject_id'] == row['subject_id']].tolist()[0]
    ) if row['subject_id'] in df_clean.loc[mask,'subject_id'].values else None
    if x_idx is None: continue
    fname = waterfall_dir / f"LOW_risk_{rank+1}_{row['subject_id']}.png"
    plot_participant_waterfall(x_idx, row['subject_id'], row['AD_risk_prob'], row['risk_tier'], fname)

print("\nSTEP 24 COMPLETE — Waterfall plots saved to:", waterfall_dir)
print("\n=== ORION V4 PACKAGING COMPLETE ===")
print(f"Package folder : {PACKAGE_DIR}")
print(f"Risk CSV       : {risk_csv}")
print(f"Waterfall plots: {waterfall_dir}")

# STEP 25 -- SENSITIVITY MODELS
# FIX 5: baseline_only  = no 12/24-month MMSE slopes (those use visits AFTER baseline)
# FIX 6: non_cognitive  = no MMSE, MoCA or FAQ (those feed the ADNI diagnosis itself)

print("\nSTEP 25 -- SENSITIVITY MODELS (nested CV + held-out test set)")
sensitivity_sets = {
    'baseline_only': [f for f in primary_features if f not in ('cognitive_slope', 'cognitive_slope_24')],
    'non_cognitive': ['age', 'gender', 'education_years', 'BMI', 'APOE4',
                      'APOE4_x_age', 'age_x_gender', 'education_x_APOE4'],
}
for set_name, feats in sensitivity_sets.items():
    Xs = X[feats + list(hypertension.columns)]
    search_s = RandomizedSearchCV(RandomForestClassifier(class_weight='balanced', random_state=42),
                                  param_distributions=param_grid, n_iter=30, cv=5,
                                  scoring='roc_auc', random_state=42, n_jobs=1)
    rf_nested = cross_val_score(search_s, Xs, Y, cv=cv, scoring='roc_auc')
    lr_cv     = cross_val_score(LogisticRegression(max_iter=5000, class_weight='balanced'),
                                Xs, Y, cv=cv, scoring='roc_auc')
    search_s.fit(Xs.loc[X_train.index], Y_train)
    rf_test = auc(*roc_curve(Y_test, search_s.predict_proba(Xs.loc[X_test.index])[:, 1])[:2])
    lr_fit  = LogisticRegression(max_iter=5000, class_weight='balanced').fit(Xs.loc[X_train.index], Y_train)
    lr_test = auc(*roc_curve(Y_test, lr_fit.predict_proba(Xs.loc[X_test.index])[:, 1])[:2])
    print(f"\n{set_name} ({len(Xs.columns)} features)")
    print(f"  RF nested CV AUC : {round(rf_nested.mean(),3)} +/- {round(rf_nested.std(),3)}   Test AUC: {round(rf_test,3)}")
    print(f"  LR 5-fold CV AUC : {round(lr_cv.mean(),3)} +/- {round(lr_cv.std(),3)}   Test AUC: {round(lr_test,3)}")

# STEP 26 — GENERATE STREAMLIT CLINICIAN APP
# Writes orion_app.py to your Downloads folder
# Run it with: streamlit run ~/Downloads/orion_app.py
#

streamlit_code = '''
import streamlit as st
import pickle, numpy as np, pandas as pd
import shap, matplotlib.pyplot as plt
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier as _SkRF
from sklearn.linear_model import LogisticRegression as _SkLR
class _TrainOnlyImpute:
    def impute(self, X):
        return pd.DataFrame(X).pipe(lambda d: d.mask(d.isna(), self.fill_, axis=1))
    def predict(self, X):
        return super().predict(self.impute(X))
    def predict_proba(self, X):
        return super().predict_proba(self.impute(X))
class RandomForestClassifier(_TrainOnlyImpute, _SkRF): pass
class LogisticRegression(_TrainOnlyImpute, _SkLR):
    def decision_function(self, X):
        return super().decision_function(self.impute(X))

# ── Load packaged model 
PACKAGE = Path.home() / "Downloads" / "ORION_v4_package"

@st.cache_resource
def load_artifacts():
    with open(PACKAGE / "rf_tuned.pkl",        "rb") as f: model    = pickle.load(f)
    with open(PACKAGE / "feature_columns.pkl", "rb") as f: features = pickle.load(f)
    return model, features

model, features = load_artifacts()
explainer = shap.TreeExplainer(model)

# ── Page config
st.set_page_config(page_title="ORION v4 — AD Risk", page_icon="🧠", layout="centered")
st.title("🧠 ORION v4 — Alzheimer's Risk Predictor")
st.caption("Research use only. Not for clinical diagnosis.")
st.divider()

# ── Patient input form
st.subheader("Enter Patient Details")

col1, col2 = st.columns(2)

with col1:
    age            = st.number_input("Age (years)",              min_value=50,  max_value=95,  value=72)
    gender         = st.selectbox("Sex",                         ["Male (0)", "Female (1)"])
    education      = st.number_input("Education (years)",        min_value=0,   max_value=25,  value=14)
    bmi            = st.number_input("BMI",                      min_value=15.0,max_value=45.0,value=26.0)
    apoe4          = st.selectbox("APOE4 Carrier",               ["No (0)", "Yes (1)"])
    mmse           = st.number_input("MMSE Score",               min_value=0,   max_value=30,  value=26)

with col2:
    moca           = st.number_input("MoCA Score",               min_value=0,   max_value=30,  value=24)
    moca_real      = st.selectbox("MoCA — is this a real score?",["Yes (1)", "No / imputed (0)"])
    iadl           = st.number_input("FAQ / IADL Total Score",   min_value=0,   max_value=30,  value=3)
    cog_slope      = st.number_input("Cognitive Slope 12m (MMSE change)", min_value=-10.0, max_value=5.0, value=0.0, step=0.5)
    cog_slope_24   = st.number_input("Cognitive Slope 24m (MMSE change)", min_value=-15.0, max_value=5.0, value=0.0, step=0.5)
    hypertension   = st.selectbox("Hypertension",                ["No (0)", "Yes (1)"])

st.divider()

# ── Parse inputs 
gender_val     = 0 if "Male"   in gender    else 1
apoe4_val      = 1 if "Yes"    in apoe4     else 0
moca_real_val  = 1 if "Yes"    in moca_real else 0
hypert_val     = 1 if "Yes"    in hypertension else 0

# Interaction terms (must match training exactly)
apoe4_x_age       = apoe4_val   * age
age_x_gender      = age         * gender_val
mmse_x_age        = mmse        * age
iadl_x_gender     = iadl        * gender_val
education_x_apoe4 = education   * apoe4_val

input_dict = {
    "age"              : age,
    "gender"           : gender_val,
    "education_years"  : education,
    "BMI"              : bmi,
    "APOE4"            : apoe4_val,
    "mmse_score"       : mmse,
    "moca_score"       : moca,
    "moca_real"        : moca_real_val,
    "iadl_score"       : iadl,
    "cognitive_slope"  : cog_slope,
    "cognitive_slope_24": cog_slope_24,
    "APOE4_x_age"      : apoe4_x_age,
    "age_x_gender"     : age_x_gender,
    "mmse_x_age"       : mmse_x_age,
    "iadl_x_gender"    : iadl_x_gender,
    "education_x_APOE4": education_x_apoe4,
    "hypertension_1"   : hypert_val,
}

# Align to exact feature columns the model was trained on
input_df = pd.DataFrame([input_dict])
for col in features:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[features]

# ── Predict 
if st.button("Calculate AD Risk", type="primary", use_container_width=True):

    prob    = model.predict_proba(input_df)[0][1]
    percent = f"{prob:.1%}"

    if prob >= 0.66:
        tier, color, emoji = "HIGH",     "#dc2626", "🔴"
    elif prob >= 0.33:
        tier, color, emoji = "MODERATE", "#d97706", "🟡"
    else:
        tier, color, emoji = "LOW",      "#16a34a", "🟢"

    # Risk score display
    st.markdown(f"""
    <div style="background:{color}18; border-left:5px solid {color};
                border-radius:10px; padding:20px 24px; margin:16px 0;">
      <div style="font-size:13px; color:#64748b; font-weight:600;">AD RISK PROBABILITY</div>
      <div style="font-size:48px; font-weight:800; color:{color};">{emoji} {percent}</div>
      <div style="font-size:16px; font-weight:600; color:{color};">{tier} RISK</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # SHAP waterfall for this patient
    st.subheader("Contributing Risk Factors")
    st.caption("Red bars push risk UP. Blue bars push risk DOWN. Width = strength of influence.")

    shap_vals = explainer(input_df)
    fig, ax = plt.subplots(figsize=(9, 5))
    shap.plots.waterfall(shap_vals[:, :, 1][0], max_display=14, show=False)
    st.pyplot(plt.gcf(), use_container_width=True)
    plt.close()

    st.divider()
    st.caption("⚠️ ORION v4 is a research tool. Results must be interpreted by a qualified clinician.")
'''

app_path = original_path + "orion_app.py"
with open(app_path, "w") as f:
    f.write(streamlit_code)

print("\nSTEP 26 COMPLETE")
print("Streamlit app written to:", app_path)
print("\nTo launch it, open Terminal and run:")
print("  pip install streamlit")
print("  streamlit run ~/Downloads/orion_app.py")

# STEP 27 — AUTO-LAUNCH STREAMLIT APP

import subprocess, sys

print("\nSTEP 27 -- Launching clinician app...")
print("Opening in browser at http://localhost:8501")
print("Press Control+C in Terminal to stop the app when done.")

subprocess.run([sys.executable, "-m", "streamlit", "run", 
                original_path + "orion_app.py"])
