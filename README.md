# ORION: Interpretable Machine Learning for Alzheimer's Disease Classification
Interpretable machine learning for Alzheimer’s disease classification from routine clinical, functional, and genetic data (ADNI). Code for the ORION manuscript and the Neurobility clinician dashboard.


## Contents
- `v4ORADPRED.py`: full analysis pipeline (data preparation, nested cross-validation, sensitivity analyses, SHAP explanations, model packaging).
- `neurobility_dashboard.py`: Streamlit clinician prototype that loads the trained model and returns a patient's risk probability, risk tier, and SHAP explanation. Research use only.


## Data

This study used data from the Alzheimer's Disease Neuroimaging Initiative (ADNI). ADNI data cannot be redistributed. Qualified researchers can apply for access at https://adni.loni.usc.edu.

The script expects four CSV files. Set the two folder paths at the top of `v4ORADPRED.py` (`original_path` and `new_path`) to where you saved them.

**In `original_path`:**
- `all_subject_AD_ORION1_My_Table_02Mar2026.csv`: merged participant-level table (one row per visit), downloaded from ADNI on March 2, 2026. Required columns: `subject_id`, `visit`, `entry_age`, `PTGENDER`, `PTEDUCAT`, `GENOTYPE`, `MMSCORE`, `VSWEIGHT`, `VSHEIGHT`, `MH4CARD`, `DXNORM`, `DXMCI`, `DXAD`, `DXPARK`, `DXOTHDEM`.

**In `new_path`:**
- `All_Subjects_MOCA_22Mar2026.csv`: ADNI MoCA table. Columns used: `PTID`, `VISCODE`, `MOCA`.
- `All_Subjects_FAQ_22Mar2026.csv`: ADNI Functional Activities Questionnaire table. Columns used: `PTID`, `VISCODE`, `FAQTOTAL`.
- `All_Subjects_DXSUM_22Mar2026.csv`: ADNI diagnostic summary table. Columns used: `PTID`, `VISCODE`, `DXPARK`, `DXOTHDEM`.

If your files have different names, update the filenames in Step 2 of the script.
