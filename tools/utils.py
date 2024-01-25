from typing import Dict

import pandas as pd
import re

INFLAMMATION_LAB_TESTS = [
    51301,  # "White Blood Cells",
    51755,
    51300,
    50889,  # "C-Reactive Protein"
    51652,
]


ADDITIONAL_LAB_TEST_MAPPING = {
    "Complete Blood Count (CBC)": [
        51279,  # "Red Blood Cells",
        51301,  # "White Blood Cells",
        51755,
        51300,
        51222,  # "Hemoglobin",
        50811,
        51221,  # "Hematocrit",
        51638,
        50810,
        51250,  # "MCV",
        51248,  # "MCH",
        51249,  # "MCHC",
        51265,  # "Platelet Count",
        51244,  # "Lymphocytes",
        51133,  # "Absolute Lymphocyte Count",
        52769,
        51245,  # "Lymphocytes, Percent",
        51146,  # "Basophils",
        52069,  # "Absolute Basophil Count",
        51200,  # "Eosinophils",
        52073,  # "Absolute Eosinophil Count",
        51254,  # "Monocytes",
        52074,  # "Absolute Monocyte Count",
        51253,  # "Monocyte Count"
        51256,  # "Neutrophils",
        52075,  # "Absolute Neutrophil Count",
        51277,  # "RDW",
        52172,  # "RDW-SD",
    ],
    "Basic Metabolic Panel (BMP)": [
        50809,  # "Glucose",
        50931,
        52569,
        50824,  # "Sodium",
        50983,
        52623,
        50822,  # "Potassium",
        50971,
        52610,
        50806,  # "Chloride",
        50902,
        52535,
        50803,  # "Bicarbonate",
        50882,
        51006,  # "Urea Nitrogen",
        52647,
        50912,  # "Creatinine",
        52024,
        52546,
        50808,  # "Calcium",
        50893,
        51624,
    ],
    "Comprehensive Metabolic Panel (CMP)": [
        50809,  # "Glucose",
        50931,
        52569,
        50824,  # "Sodium",
        50983,
        52623,
        50822,  # "Potassium",
        50971,
        52610,
        50806,  # "Chloride",
        50902,
        52535,
        50803,  # "Bicarbonate",
        50882,
        51006,  # "Urea Nitrogen",
        52647,
        50912,  # "Creatinine",
        52024,
        52546,
        50808,  # "Calcium",
        50893,
        51624,
        50861,  # "Alanine Aminotransferase (ALT)",
        50863,  # "Alkaline Phosphatase",
        50878,  # "Asparate Aminotransferase (AST)",
        50883,  # "Bilirubin",
        50884,
        50885,
        50976,  # "Total Protein",
    ],
    "Blood urea nitrogen (BUN)": [
        51006,  # "Urea Nitrogen",
        52647,
    ],
    "Renal Function Panel (RFP)": [
        50862,  # "Albumin",
        51006,  # "Urea Nitrogen",
        52647,
        50824,  # "Sodium",
        50983,
        52623,
        50808,  # "Calcium",
        50893,
        51624,
        50804,  # "C02",
        51739,
        50806,  # "Chloride",
        50902,
        52535,
        50912,  # "Creatinine",
        52024,
        52546,
        50809,  # "Glucose",
        50931,
        52569,
        50970,  # "Phosphate",
        50822,  # "Potassium",
        50971,
        52610,
    ],
    "Liver Function Panel (LFP)": [
        50861,  # "Alanine Aminotransferase (ALT)",
        50878,  # "Asparate Aminotransferase (AST)",
        50863,  # "Alkaline Phosphatase",
        50927,  # "Gamma Glutamyltransferase",
        50883,  # "Bilirubin",
        50884,
        50885,
        51274,  # "Prothrombin Time (PT)",
        51237,
        51675,
        50976,  # "Total Protein",
        50862,  # "Albumin",
    ],
    "Urinalysis": [
        51508,  # "Urine Color",
        51506,  # "Urine Appearance",
        51512,  # "Urine Mucous",
        51108,  # "Urine Volume",
        51498,  # "Specific Gravity",
        51994,
        51093,  # "Osmolality, Urine",
        51069,  # "Albumin, Urine",
        51070,  # "Albumin/Creatinine, Urine",
        51082,  # "Creatinine, Urine",\
        51106,
        51097,  # "Potassium, Urine",
        51100,  # "Sodium, Urine",
        51102,  # "Total Protein, Urine",
        51068,
        51492,
        51992,
        51104,  # "Urea Nitrogen, Urine",
        51462,  # "Amorphous Crystals",
        51469,  # "Calcium Oxalate Crystals",
        51503,  # "Triple Phosphate Crystals",
        51505,  # "Uric Acid Crystals",
        51510,  # "Urine Crystals, Other",
        51493,  # "RBC",
        51494,  # "RBC Casts",
        51495,  # "RBC Clumps",
        51516,  # "WBC",
        51517,  # "WBC Casts",
        51518,  # "WBC Clumps",
        51507,  # "Urine Casts, Other",
        51094,  # "pH"
        51491,
        52730,
        51464,  # "Bilirubin",
        51966,
        51084,  # "Glucose",
        51478,
        51981,
        51514,  # "Urobilinogen",
        52002,
        51484,  # "Ketone",
        51984,
        51487,  # "Nitrite",
        51987,
        51486,  # "Leukocytes",
        51985,
        51476,  # "Epithelial Cells",
        51497,
        51501,
        51489,
        51488,
    ],
    "Electrolyte Panel": [
        50824,  # "Sodium",
        50983,
        52623,
        50822,  # "Potassium",
        50971,
        52610,
        50806,  # "Chloride",
        50902,
        52535,
        50803,  # "Bicarbonate",
        50882,
    ],
    "Lipid Profile": [
        50907,  # "Cholesterol, Total",
        50905,  # "Cholesterol, LDL, Calculated",
        50906,  # "Cholesterol, LDL, Measured",
        50904,  # "Cholesterol, HDL",
        51000,  # "Triglycerides",
    ],
    "Coagulation Profile": [
        51274,  # "PT",
        51275,  # "PTT",
        51675,  # "INR(PT)",
        51237,  # "INR(PT)",
        # aPTT and TT not found
    ],
    "Iron Studies": [
        50952,  # "Iron",
        50953,  # "Iron Binding Capacity, Total"
        50998,  # "Transferrin",
        50924,  # "Ferritin",
        51250,  # "MCV",
        # TSAT not found
    ],
    "Liver Enzymes": [
        50861,  # "Alanine Aminotransferase (ALT)",
        50878,  # "Asparate Aminotransferase (AST)",
        50863,  # "Alkaline Phosphatase",
        50927,  # "Gamma Glutamyltransferase",
    ],
    "Thyroid Function Test (TFT)": [
        50993,  # "Thyroid Stimulating Hormone",
        50994,  # "Thyroxine (T4)",
        50995,  # "Thyroxine (T4), Free",
        51001,  # "Triiodothyronine (T3)",
        50992,  # "Thyroid Peroxidase Antibody",
    ],
    "Gamma Glutamyltransferase (GGT)": [
        50927,  # "Gamma Glutamyltransferase",
    ],
    "Phosphorus": [
        50970,  # "Phosphate",
    ],
    "Mean Corpuscular Volume (MCV)": [
        51250,  # "MCV",
    ],
    "C-Reactive Protein (CRP)": [
        50889,  # "C-Reactive Protein"
    ],
    "CRP": [
        50889,  # "C-Reactive Protein"
    ],
    "Prostate Specific Antigen (PSA)": [
        50974,  # ["Prostate Specific Antigen"],
    ],
    "PSA": [
        50974,  # ["Prostate Specific Antigen"],
    ],
    "Alkaline Phosphatase (ALP)": [
        50863,  # "Alkaline Phosphatase",
    ],
    "ALP": [
        50863,  # "Alkaline Phosphatase",
    ],
    "Alk Phos": [
        50863,  # "Alkaline Phosphatase",
    ],
    "Pregnancy Test": [
        51085,  # "HCG, Urine, Qualitative",
        52720,
    ],
    "Amylase, Serum": [
        50867,  # "Amylase",
    ],
    "Serum Amylase": [
        50867,  # "Amylase",
    ],
    "Stool Occult Blood": [
        51460,  # "Occult Blood",
    ],
    "Erythrocyte Sedimentation Rate (ESR)": [
        51288,  # "Sedimentation Rate",
    ],
    "ESR": [
        51288,  # "Sedimentation Rate",
    ],
    "Troponin": [
        51003,  # "Troponin T",
    ],
    "Direct Bilirubin": [
        50883,  # "Bilirubin, Direct",
    ],
    "TSH": [
        50993,  # "Thyroid Stimulating Hormone",
    ],
    "Calcium": [
        50893,  # "Calcium, Total",
    ],
    "White Blood Cell Count": [
        51300,  # "WBC Count",
    ],
    "Free T4": [
        50995,  # "Thyroxine (T4), Free",
    ],
    "Blood Culture": [
        90201,  # "Blood Culture, Routine"
    ],
    "Stool Culture": [
        90267,  # "Stool Culture"
    ],
    "O&P": [
        90250,  # "OVA + PARASITES"
    ],
    # Procalcitonin not found
}

ADDITIONAL_LAB_TEST_MAPPING_SYNONYMS = {
    "Renal Function Test (RFT)": "Renal Function Panel (RFP)",
    "Renal Function Test panel": "Renal Function Panel (RFP)",
    "Renal Function": "Renal Function Panel (RFP)",
    "Kidney Function Tests (KFT)": "Renal Function Panel (RFP)",
    "KFTs": "Renal Function Panel (RFP)",
    "Liver Function Test (LFT)": "Liver Function Panel (LFP)",
    "Liver Function Test panel": "Liver Function Panel (LFP)",
    "LFTs": "Liver Function Panel (LFP)",
    "Urine Analysis (UA)": "Urinalysis",
    "Electrolytes": "Electrolyte Panel",
    "Serum Electrolytes": "Electrolyte Panel",
    "Urine Pregnancy Test": "Pregnancy Test",
}

LAB_TEST_MAPPING_SYNONYMS = {
    51300: 51301,  # "WBC Count": "White Blood Cells",
    50810: 51221,  # Hematocrit, calculated: Hematocrit
    51108: 51109,  # "Urine Volume, Total": "Urine Volume",
    51237: 51274,  # "INR(PT)": "PT",
    51068: 51492,  # "24 hr Protein": "Protein"
    51102: 51492,  # "Total Protein, Urine": "Protein"
    51084: 51478,  # Glucose, Urine: Glucose
    51488: 51476,  # epithelial cells
    51489: 51476,  # epithelial cells
    51497: 51476,  # epithelial cells
    51501: 51476,  # epithelial cells
    51277: 52172,  # RDW: RDW-SD
    50824: 50983,  # Sodium, Whole Blood : Sodium
    50822: 50971,  # Potassium, Whole Blood : Potassium
    50806: 50902,  # Chloride, Whole Blood : Chloride
    50803: 50882,  # Bicarbonate, Whole Blood : Bicarbonate
    52024: 50912,  # Creatinine, Whole Blood : Creatinine
    50883: 50885,  # Bilirubin
    50884: 50885,  # Bilirubin
    51082: 51106,  # Creatinine, Urine : Urine Creatinine
    52069: 51146,  # Absolute Basophil Count: Basophils
    51133: 51244,  # Absolute Lymphocyte Count: Lymphocytes
    52073: 51200,  # Absolute Eosinophil Count: Eosinophils
    52074: 51254,  # Absolute Monocyte Count: Monocytes
    52075: 51256,  # Absolute Neutrophil Count: Neutrophils
}

LAB_TEST_MAPPING_ALTERATIONS = {
    "Treponema pallidum (Syphilis) Ab": "Treponema pallidum, syphilis, Ab",
    "Treponema pallidum (syphilis) value": "Treponema pallidum, syphilis, value",
    "Gray Top Hold (plasma)": "Gray Top Hold, plasma",
    "Green Top Hold (plasma)": "Green Top Hold, plasma",
    "K (GREEN)": "K, GREEN",
}

FLUID_MAPPING = {
    "Blood": ["Blood", "Plasma", "Serum"],
    "Urine": [
        "Urine",
    ],
    "Stool": [
        "Stool",
    ],
    "Ascites": [
        "Ascites",
    ],
    "Cerebrospinal Fluid": [
        "Cerebrospinal Fluid",
    ],
    "Joint Fluid": [
        "Joint Fluid",
    ],
    "Bone Marrow": [
        "Bone Marrow",
    ],
    "Pleural": [
        "Pleural",
    ],
    "Other Body Fluid": [
        "Other Body Fluid",
    ],
}


REGION_EXACT_DICT = {"Abdomen": ["gi", "eus", "mrcp", "hida", "ercp"]}

REGION_SUBSTR_DICT = {
    "Chest": [
        "chest",
        "lung",
        "upper lobe",
        "lower lobe",
        "pleura",
        "atelectasis",
        "ground.glass",
        "heart",
        "cardiac",
        "pericard",
        "mediastin",
        "pneumothorax",
        "breast",
    ],
    "Abdomen": [
        "abd",
        "abdom",
        "pelvi",
        "liver",
        "gallbladder",
        "pancrea",
        "duct",
        "spleen",
        "stomach",
        "bowel",
        "rectum",
        "ileum",
        "iliac",
        "duodenum",
        "colon",
        "urinary",
        "bladder",
        "ureter",
        "kidney",
        "renal",
        "adrenal glands",
        "intraperitoneal",
        "ascites",
        "prostate",
        "uterus",
        "appendi",
        "retroperit",
        "mesenter",
        "paracolic",
        "lower quadrant",
        "perirectal",
        "cul-de-sac",
        "iliopsoas",
        "psoas",
        "hepatic",
        "hepato",
        "quadrant",
        "gastro",
    ],
    "Venous": [
        "venous",
        "jugular",
        "cephalic",
        "axillary",
        "basilic",
        "brachial",
        "popliteal",
        "femoral",
        "peroneal",
        "tibial",
        "fibular",
        "veins",
    ],
    "Head": ["head", "brain", "skull"],
    "Neck": ["neck", "thyroid"],
    "Scrotum": ["scrot", "testic"],
    "Spine": ["spine", "cervical"],
    "Ankle": ["ankle"],
    "Foot": ["foot"],
    "Bone": ["bone"],
    "Knee": ["knee"],
    "Hand": ["hand"],
    "Wrist": ["wrist"],
    "Finger": ["finger"],
    "Heel": ["heel"],
    "Hip": ["hip"],
    "Shoulder": ["shoulder"],
    "Thigh": ["thigh"],
    "Femur": ["femur"],
    #'Extremity' : ['extremity', 'arm', 'leg', 'thigh', 'knee', 'hands']
    # "Upper Extremity": ["up(.*)ext"],
    # "Lower Extremity": ["low(.*)ext"],
}

MODALITY_EXACT_DICT = {
    "CT": ["ct", "cat", "cta", "mdct", "ctu", "dlp", "mgy"],
    "Ultrasound": ["us", "dup"],
    "Radiograph": ["ap", "pa", "cxr"],
    "MRI": ["mri", "mr", "mrcp", "t\d"],
}

MODALITY_SUBSTR_DICT = {
    "CT": ["multidetector", "reformat", "optiray"],
    "Ultrasound": [
        "u\.s\.",
        "ultrasound",
        "echotexture",
        "sonogra",
        "doppler",
        "duplex",
        "doppler",
        "echogenic",
        "transabdominal",
        "transvaginal",
        "non-obstetric",
    ],
    "Radiograph": [
        "port\.",
        "radiograph",
        "portable",
        "x-ray",
        "supine and",
        "supine &",
        "supine only",
        "and lateral",
        "frontal view",
        "supine view",
        "single view",
        "two views",
    ],
    "MRI": ["gadavist", "magnet", "tesla"],
    "Fluoroscopy": ["fluoro"],
}

MODALITY_SPECIAL_CASES_DICT = {
    "CTU": [
        "ctu",
        "ct urogram",
        "ct urography",
        "ct ivu",
        "ct ivp",
        "ct intravenous pyelography",
    ],
    "Drainage": ["drain"],
    "Carotid ultrasound": ["carotid.*ultrasound", "carotid.*us", "carotid.*series"],
    "EUS": ["eus", "endoscopic.*(ultrasound|us)", "echo.*endoscopy"],
    "MRCP": ["mrcp", "magnetic.*resonance.*cholangiopancreatography"],
    "HIDA": ["hida", "hepatobiliary.*iminodiacetic.*acid"],
    "ERCP": [
        "ercp",
        "endoscopic.*retrograde.*cholangiopancreatography",
        "bil endoscopy",
    ],
    "PTC": [
        "ptc",
        "percutaneous.*transhepatic.*cholangiography",
        "perc transhepatic cholangiography",
    ],
    "Upper GI Series": [
        "upper.*gi",
        "upper.*gastrointestinal",
        "barium.*swallow",
        "barium.*meal",
        "barium.*study",
        "ugis",
        "bas/ugi",
    ],
    "Paracentesis": ["paracentesis"],
    "Mammogram": ["mammo"],
    "MRA": ["mra", "magnetic.*resonance.*angiography"],
    "MRE": ["mre", "magnetic.*resonance.*enterography", "mr enterography"],
}

UNIQUE_TO_BROAD_MODALITY = {
    "CTU": "CT",
    "Carotid ultrasound": "Ultrasound",
    "EUS": "Ultrasound",
    "MRCP": "MRI",
    "ERCP": "Radiograph",
    "Upper GI Series": "Radiograph",
    "MRA": "MRI",
    "MRE": "MRI",
}

UNIQUE_MODALITY_TO_ORGAN_MAPPING = {
    "CTU": "Abdomen",
    "EUS": "Abdomen",
    "MRCP": "Abdomen",
    "HIDA": "Abdomen",
    "ERCP": "Abdomen",
    "MRE": "Abdomen",
    "Upper GI Series": "Abdomen",
    "Carotid ultrasound": "Neck",
    "Mammogram": "Chest",
}


# Convert action input to string
def action_input_pretty_printer(obj, lab_test_mapping_df: pd.DataFrame):
    # Check if set i.e. lab results
    if isinstance(obj, list):
        obj_str = []
        for itemid in obj:
            # Convert itemids to str
            if isinstance(itemid, int):
                obj_str.append(itemid_to_field(itemid, "label", lab_test_mapping_df))
            # Not found strs can be appended directly
            elif isinstance(itemid, str):
                obj_str.append(itemid)
            else:
                print(itemid)
                raise NotImplementedError
        return ", ".join(obj_str)
    # Check if dict i.e. imaging
    elif isinstance(obj, dict):
        return ", ".join(obj.values())
    else:
        raise NotImplementedError


# Search for terms using regex and word boundries. Count number of matches
def count_matches(
    text, exact_dict: Dict = {}, substr_dict: Dict = {}, special_cases_dict: Dict = {}
):
    counts = {}

    # If there is a special cases match, return that
    for category, patterns in special_cases_dict.items():
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if len(matches) > 0:
                counts[category] = 1
                return counts

    # Initialize counts for all categories
    for cat in set(list(exact_dict.keys()) + list(substr_dict.keys())):
        counts[cat] = 0

    # Count exact matches
    for category, words in exact_dict.items():
        for word in words:
            pattern = r"\b" + word + r"\b"
            matches = re.findall(pattern, text, re.IGNORECASE)
            counts[category] += len(matches)

    # Count substring matches
    for category, substrings in substr_dict.items():
        for pattern in substrings:
            matches = re.findall(pattern, text, re.IGNORECASE)
            counts[category] += len(matches)

    return counts


def count_radiology_modality_and_organ_matches(text):
    modality_counts = count_matches(
        text,
        exact_dict=MODALITY_EXACT_DICT,
        substr_dict=MODALITY_SUBSTR_DICT,
        special_cases_dict=MODALITY_SPECIAL_CASES_DICT,
    )
    frequent_modality = max(modality_counts, key=modality_counts.get)
    frequent_modality_count = modality_counts[frequent_modality]

    # Count matches of each region
    organ_counts = count_matches(
        text,
        exact_dict=REGION_EXACT_DICT,
        substr_dict=REGION_SUBSTR_DICT,
    )
    frequent_region = max(organ_counts, key=organ_counts.get)
    frequent_region_count = organ_counts[frequent_region]

    # if no region is found, check if a unique modality was given where the region is known
    if frequent_region_count == 0:
        if frequent_modality in UNIQUE_MODALITY_TO_ORGAN_MAPPING:
            frequent_region = UNIQUE_MODALITY_TO_ORGAN_MAPPING[frequent_modality]
            frequent_region_count = 1

    return (
        frequent_modality,
        frequent_modality_count,
        frequent_region,
        frequent_region_count,
    )


def itemid_to_field(itemid: int, field: str, lab_test_mapping_df: pd.DataFrame):
    return lab_test_mapping_df.loc[lab_test_mapping_df["itemid"] == itemid, field].iloc[
        0
    ]
