def extract_diagnosis_from_diag_df(hadm_info, diag_df):
    for _id in hadm_info:
        diagnoses = diag_df[diag_df["hadm_id"] == _id]["long_title"].values
        hadm_info[_id]["ICD Diagnosis"] = diagnoses.tolist()
    return hadm_info
