patient_x = {
    "Patient History": " ___ presents with 4 days of RLQ pain. Says symptoms started after heavy dinner. Reports decreased appetite and chills.  Past Medical History: PMH: none  PSH: none  ___: none    Social History: ___ Family History: Fam Hx: no history of Crohn's or UC. Grandfather with ___ types of cancers otherwise no hx of malignancy    ",
    "Physical Examination": " Temp: 97.8 HR: 44 BP: 104/69 RR: 17 100% Ra Gen: NAD HEENT: non icteric, atraumatic CV: RRR no m,r,g RESP: CTABL Abd: soft, non tender, non distended, incisions c/d/i Ext: wwpx4, palpable distal pulses",
    "Laboratory Tests": {
        50861: "18.0 IU/L",  # ALT
        51279: "4.74 m/uL",  # Red Blood Cells
        51301: "7.6 K/uL",  # White Blood Cells
    },
    "Microbiology": {90201: "No growth."},
    "Reference Range Lower": {50861: float("nan"), 51279: 4.2, 51301: 4.5},
    "Reference Range Upper": {50861: float("nan"), 51279: 5.9, 51301: 11.0},
    "Radiology": [
        {
            "Report": "CT Abd & Pelvis:\nFindings are compatible with previous report:\nBlind ending, 8.8 mm tubular structure rising near the base of the cecum, most likely representing the appendix, with indications of inflammation in surroundings.\nEvidence of infection of lymph nodes.",
            "Modality": "CT",
            "Region": "Abdomen",
        }
    ],
    "Discharge Diagnosis": "Diagnosis:\nacute appendicitis \n\n \n",
    "ICD Diagnosis": [
        "Acute appendicitis without mention of peritonitis",
        "Other and unspecified diseases of appendix",
    ],
    "Procedures": [],
    "Procedures ICD9": ["4701"],
    "Procedures ICD9 Title": ["Laparoscopic appendectomy"],
    "Procedures ICD10": [],
    "Procedures ICD10 Title": [],
}
