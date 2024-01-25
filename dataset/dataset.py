import warnings
from os.path import join
import re
import traceback
from collections import Counter
from datetime import timedelta

import pandas as pd

from dataset.discharge import (
    extract_history,
    extract_physical_examination,
    extract_diagnosis_from_discharge,
    extract_chief_complaints,
)
from dataset.radiology import (
    extract_rad_events,
    sanitize_rad,
)
from dataset.labs import parse_lab_events, parse_microbio
from dataset.procedures import extract_procedures
from dataset.diagnosis import extract_diagnosis_from_diag_df
from dataset.utils import write_hadm_to_file, print_value_counts
from tools.utils import count_radiology_modality_and_organ_matches


warnings.filterwarnings("default", category=UserWarning)


def extract_hadm_ids(pathology, diag_icd, discharge_df, diag_counts=20, cc=10):
    # Grab all hadm_ids with appendicitis
    hadm_ids = (
        diag_icd[diag_icd["long_title"].str.contains(pathology, case=False)][
            ["hadm_id"]
        ]
        .drop_duplicates()["hadm_id"]
        .values
    )
    print("There are {} hadm_ids with {}".format(len(hadm_ids), pathology))

    # Get appendicitis diagnoses counts
    v_counts = diag_icd[diag_icd["long_title"].str.contains(pathology, case=False)][
        "long_title"
    ].value_counts()
    print_value_counts(v_counts, diag_counts)
    print("---")

    # Get chief complaints of cholecystitis patients
    app_cc, app_cc_ids, discharge_cntr = extract_chief_complaints(
        hadm_ids, discharge_df
    )
    print(
        "There are {} patients with {} with discharge notes.".format(
            discharge_cntr, pathology
        )
    )
    app_cc = [c.lower().strip() for c in app_cc]

    # Count frequency of chief complaints
    counter = Counter(app_cc)

    # Get most common chief complaints
    print("---")
    print("Most common chief complaints:")
    for c, cnt in counter.most_common(cc):
        print("{}: {}".format(c, cnt))

    return hadm_ids


def extract_hadm_ids_filter_cc(
    pathology,
    diag_icd,
    discharge_df,
    chief_complaint="abdominal pain",
    diag_counts=20,
    cc=10,
):
    # Grab all hadm_ids with patho
    hadm_ids = (
        diag_icd[diag_icd["long_title"].str.contains(pathology, case=False)][
            ["hadm_id"]
        ]
        .drop_duplicates()["hadm_id"]
        .values
    )
    print("There are {} hadm_ids with {}".format(len(hadm_ids), pathology))

    # Get appendicitis diagnoses counts
    v_counts = diag_icd[diag_icd["long_title"].str.contains(pathology, case=False)][
        "long_title"
    ].value_counts()
    print_value_counts(v_counts, diag_counts)
    print("---")

    # Get chief complaints of cholecystitis patients
    app_cc, app_cc_ids, discharge_cntr = extract_chief_complaints(
        hadm_ids, discharge_df
    )
    print(
        "There are {} patients with {} with discharge notes.".format(
            discharge_cntr, pathology
        )
    )
    app_cc = [c.lower().strip() for c in app_cc]

    filtered_ids = []
    for indx, cc in enumerate(app_cc):
        if chief_complaint == cc.lower():
            filtered_ids.append(app_cc_ids[indx])
    print("{} patients who presented with abdominal pain".format(len(filtered_ids)))

    return filtered_ids


def extract_info(
    hadm_ids,
    pathology,
    sanitize_list,
    discharge_df,
    admissions_df,
    transfers_df,
    lab_events_df,
    microbiology_df,
    radiology_report_df,
    radiology_report_details_df,
    diag_df,
    procedures_df,
):
    # Extract the discharge, history, pe, le and radiology report for hadm_ids
    hadm_info = extract_hadm_info(
        hadm_ids,
        discharge_df,
        admissions_df,
        transfers_df,
        lab_events_df,
        microbiology_df,
        radiology_report_df,
        radiology_report_details_df,
    )
    print("--")

    hadm_info_clean = None
    try:
        # Fill extracted reports into app_hadm_info
        # hadm_info = chatgpt_extractor(hadm_info, pathology)
        # print("--")

        # Remove rad reports where no rad_modality was found
        hadm_info = sanitize_rad(hadm_info)
        print("--")

        # Remove mentions of target
        hadm_info = sanitize_hadm_texts(hadm_info, sanitize_list)
        print("--")

        # Extract diagnoses
        for _id in hadm_info:
            try:
                hadm_info[_id][
                    "Discharge Diagnosis"
                ] = extract_diagnosis_from_discharge(hadm_info[_id]["Discharge"])
            except Exception as e:
                print("ID: {}, Error: {}".format(_id, e))
                hadm_info[_id]["Discharge Diagnosis"] = ""
        hadm_info = extract_diagnosis_from_diag_df(hadm_info, diag_df)

        # Extract procedures
        procedures_df_icd9 = procedures_df[procedures_df["icd_version"] == 9]
        procedures_df_icd10 = procedures_df[procedures_df["icd_version"] == 10]
        hadm_info = extract_procedures(
            hadm_info, procedures_df_icd9, procedures_df_icd10
        )

        # Examine data completeness
        hadm_info_clean = check_missing(hadm_info, pathology)
        print("--")

        # Write human readable and pickle files
        write_hadm_to_file(
            hadm_info, "{}_hadm_info".format("_".join(pathology.split()))
        )
        write_hadm_to_file(
            hadm_info_clean, "{}_hadm_info_clean".format("_".join(pathology.split()))
        )
        print("Finished writing files")

    except Exception as e:
        print("Error in extracting info:, ", e)
        traceback.print_exc()

        return hadm_info, hadm_info_clean

    return hadm_info, hadm_info_clean


def create_valuestr_lab(row):
    valuenum = row["valuenum"]
    value = row["value"]
    valueuom = row["valueuom"]

    # First try to extract valuenum (Value numeric)
    if valuenum == valuenum and valuenum != "___":
        # If units of measurement is not NaN, include
        if valueuom == valueuom:
            return str(valuenum) + " " + valueuom
        return str(valuenum)

    # Next, try to extract value
    if value == value and value != "___":
        # Again check if units of measurement is not NaN but also check if value is actually number so uom makes sense
        if valueuom == valueuom:
            return str(value) + " " + valueuom
        return str(value)

    # Check if flag is not NaN (i.e. abnormal)
    if row["flag"] == row["flag"]:
        return row["flag"]

    # Finally, return comments if all else fails
    return row["comments"]


def create_valuestr_microbio(row):
    org_name = row["org_name"]
    comment = row["comments"]

    # Check if orgname present
    if org_name == org_name:
        return org_name
    else:
        return comment


def load_data(base_mimic: str = ""):
    base_hosp = join(base_mimic, "hosp")
    base_notes = join(base_mimic, "note")

    # Load admissions
    admissions_df = pd.read_csv(join(base_hosp, "admissions.csv"))

    # Load transfers
    transfers_df = pd.read_csv(join(base_mimic, "hosp", "transfers.csv"))

    diagnoses_icd_df = pd.read_csv(join(base_hosp, "diagnoses_icd.csv"))
    # remove NAN ICD Codes
    diagnoses_icd_df = diagnoses_icd_df[~diagnoses_icd_df.icd_code.isna()]

    # ICD Descriptions
    icd_descriptions = pd.read_csv(join(base_hosp, "d_icd_diagnoses.csv"))

    # Expand to include names of disease, once for version 9 and once for version 10
    diag_icd9 = diagnoses_icd_df[diagnoses_icd_df.icd_version == 9]
    icd_descriptions_9 = icd_descriptions[icd_descriptions.icd_version == 9]
    diag_icd9 = diag_icd9.merge(
        icd_descriptions_9[["icd_code", "long_title"]], on="icd_code", how="left"
    )

    diag_icd10 = diagnoses_icd_df[diagnoses_icd_df.icd_version == 10]
    icd_descriptions_10 = icd_descriptions[icd_descriptions.icd_version == 10]
    diag_icd10 = diag_icd10.merge(
        icd_descriptions_10[["icd_code", "long_title"]], on="icd_code", how="left"
    )

    diag_icd = pd.concat([diag_icd9, diag_icd10])

    # Load procedures
    procedures_df = pd.read_csv(join(base_hosp, "procedures_icd.csv"))

    # Load description of procedures and merge
    procedures_descr_df = pd.read_csv(join(base_hosp, "d_icd_procedures.csv"))
    procedures_descr_9_df = procedures_descr_df[procedures_descr_df.icd_version == 9]
    procedures_descr_10_df = procedures_descr_df[procedures_descr_df.icd_version == 10]
    procedures_9_df = procedures_df[procedures_df.icd_version == 9]
    procedures_10_df = procedures_df[procedures_df.icd_version == 10]
    procedures_9_df = procedures_9_df.merge(
        procedures_descr_9_df[["icd_code", "long_title"]], on="icd_code", how="left"
    )
    procedures_10_df = procedures_10_df.merge(
        procedures_descr_10_df[["icd_code", "long_title"]], on="icd_code", how="left"
    )
    procedures_df = pd.concat([procedures_9_df, procedures_10_df])

    # Load notes
    discharge_df = pd.read_csv(join(base_notes, "discharge.csv"))

    # Load radiology reports
    radiology_report_df = pd.read_csv(join(base_notes, "radiology.csv"))

    # Load radiology report details
    radiology_report_details_df = pd.read_csv(join(base_notes, "radiology_detail.csv"))

    # Load microbiology events
    microbiology_df = pd.read_csv(join(base_hosp, "microbiologyevents.csv"))
    # Remove canceled tests
    microbiology_df = microbiology_df[microbiology_df["org_itemid"] != 90760.0]

    # Load lab events
    lab_events_df = pd.read_csv(join(base_hosp, "labevents.csv"))

    # Load lab event descriptions
    lab_events_descr_df = pd.read_csv(join(base_hosp, "d_labitems.csv"))

    # Expand lab events to include descriptions
    lab_events_df = lab_events_df.merge(
        lab_events_descr_df[["itemid", "label"]], on="itemid", how="left"
    )

    # Create valuestr from valuenum and valueuom
    lab_events_df["valuestr"] = lab_events_df.apply(
        lambda row: create_valuestr_lab(row),
        axis=1,
    )

    # Create valuestr for microbio
    microbiology_df["valuestr"] = microbiology_df.apply(
        lambda row: create_valuestr_microbio(row),
        axis=1,
    )

    # Many lab events dont have a hadm_id. If a patient only has one hadm_id we can fill that in for them
    # get all patients with only one hadm_id
    # NOTE: This has been disabled because it will include follow-up lab results to the admission. If a test was done in the followup but never in the initial admissions, it will be included in the framework and can lead to conflicting evidence. We now select all tests between a day before admission and discharge as valid tests. This is done during the extraction of the hadm_info.
    # subjects_with_one_hadm_id = (
    #     admissions_df.groupby("subject_id").agg({"hadm_id": "nunique"}).reset_index()
    # )
    # subjects_with_one_hadm_id = subjects_with_one_hadm_id[
    #     subjects_with_one_hadm_id.hadm_id == 1
    # ]
    # subjects_with_one_hadm_id = subjects_with_one_hadm_id.subject_id.values
    # subjects_with_one_hadm_id_df = admissions_df[
    #     admissions_df.subject_id.isin(subjects_with_one_hadm_id)
    # ]
    # lab_events_df = lab_events_df.merge(
    #     subjects_with_one_hadm_id_df[["subject_id", "hadm_id"]],
    #     on="subject_id",
    #     how="left",
    # )
    # lab_events_df["hadm_id"] = lab_events_df["hadm_id_x"].fillna(
    #     lab_events_df["hadm_id_y"]
    # )
    # lab_events_df = lab_events_df.drop(columns=["hadm_id_x", "hadm_id_y"])
    # # same for microbiology
    # microbiology_df = microbiology_df.merge(
    #     subjects_with_one_hadm_id_df[["subject_id", "hadm_id"]],
    #     on="subject_id",
    #     how="left",
    # )
    # microbiology_df["hadm_id"] = microbiology_df["hadm_id_x"].fillna(
    #     microbiology_df["hadm_id_y"]
    # )
    # microbiology_df = microbiology_df.drop(columns=["hadm_id_x", "hadm_id_y"])

    # Convert transfers to datetime
    transfers_df["intime"] = pd.to_datetime(transfers_df["intime"])
    # Convert lab events charrtime to datetime
    lab_events_df["charttime"] = pd.to_datetime(lab_events_df["charttime"])
    # Convert microbiology charttime to datetime
    microbiology_df["charttime"] = pd.to_datetime(microbiology_df["charttime"])

    return (
        admissions_df,
        transfers_df,
        diag_icd,
        procedures_df,
        discharge_df,
        radiology_report_df,
        radiology_report_details_df,
        lab_events_df,
        microbiology_df,
    )


def fill_nan_hadm(
    lab_events_df,
    radiology_reports_df,
    microbiology_df,
    disease_ids,
    transfers_df,
    hadm_to_subject_id,
):
    for _id in disease_ids:
        s_id = hadm_to_subject_id[_id]

        time_data = transfers_df[transfers_df["hadm_id"] == _id].sort_values("intime")
        start_time = pd.Timestamp(time_data["intime"].values[0])
        end_time = pd.Timestamp(time_data["intime"].values[-1])

        mask_le = (
            (lab_events_df["subject_id"] == s_id)
            & (lab_events_df["charttime"] >= (start_time - timedelta(days=1)))
            & (lab_events_df["charttime"] <= end_time)
            & (lab_events_df["hadm_id"].isna())
        )
        lab_events_df.loc[mask_le, "hadm_id"] = _id

        mask_rad = (
            (radiology_reports_df["subject_id"] == s_id)
            & (radiology_reports_df["charttime"] >= (start_time - timedelta(days=1)))
            & (radiology_reports_df["charttime"] <= end_time)
            & (radiology_reports_df["hadm_id"].isna())
        )
        radiology_reports_df.loc[mask_rad, "hadm_id"] = _id

        mask_micro = (
            (microbiology_df["subject_id"] == s_id)
            & (microbiology_df["charttime"] >= (start_time - timedelta(days=1)))
            & (microbiology_df["charttime"] <= end_time)
            & (microbiology_df["hadm_id"].isna())
        )
        microbiology_df.loc[mask_micro, "hadm_id"] = _id

    return lab_events_df, radiology_reports_df, microbiology_df


def extract_hadm_info(
    disease_ids,
    discharge_df,
    admissions_df,
    transfers_df,
    lab_events_df,
    microbiology_df,
    radiology_report_df,
    radiology_report_details_df,
):
    skipped = 0
    lab_events_df["charttime"] = pd.to_datetime(lab_events_df["charttime"])
    microbiology_df["charttime"] = pd.to_datetime(microbiology_df["charttime"])
    transfers_df["intime"] = pd.to_datetime(transfers_df["intime"])
    admissions_df["admittime"] = pd.to_datetime(admissions_df["admittime"])
    admissions_df["dischtime"] = pd.to_datetime(admissions_df["dischtime"])
    radiology_report_df["charttime"] = pd.to_datetime(radiology_report_df["charttime"])

    # Create a DataFrame to hold only relevant fields with field_ordinal == 1
    filtered_radiology_report_details_df = radiology_report_details_df[
        (
            radiology_report_details_df["field_name"].isin(
                ["exam_name", "parent_note_id"]
            )
        )
        & (radiology_report_details_df["field_ordinal"] == 1)
    ]

    # Create a dictionary to map note_id to parent_note_id
    parent_note_map = (
        filtered_radiology_report_details_df[
            filtered_radiology_report_details_df["field_name"] == "parent_note_id"
        ]
        .set_index("note_id")["field_value"]
        .to_dict()
    )

    # Create a dictionary to map note_id to exam_name
    exam_name_map = (
        filtered_radiology_report_details_df[
            filtered_radiology_report_details_df["field_name"] == "exam_name"
        ]
        .set_index("note_id")["field_value"]
        .to_dict()
    )

    # Create a mask to filter out relevant rows upfront
    mask_discharge = discharge_df["hadm_id"].isin(disease_ids)

    # Filtered DataFrames
    filtered_discharge = discharge_df[mask_discharge]

    # Create dictionaries for fast lookup
    discharge_dict = filtered_discharge.set_index("hadm_id").to_dict(orient="index")

    # Create dict of hadm to subject_id
    hadm_to_subject_id = (
        admissions_df[["hadm_id", "subject_id"]]
        .set_index("hadm_id")
        .to_dict()["subject_id"]
    )

    possible_subject_ids = admissions_df[admissions_df["hadm_id"].isin(disease_ids)][
        "subject_id"
    ].unique()
    lab_events_df_sf = lab_events_df[
        lab_events_df["subject_id"].isin(possible_subject_ids)
    ].copy()
    microbiology_df_sf = microbiology_df[
        microbiology_df["subject_id"].isin(possible_subject_ids)
    ].copy()
    radiology_report_df_sf = radiology_report_df[
        radiology_report_df["subject_id"].isin(possible_subject_ids)
    ].copy()

    # Drop NaN values
    print(
        "Dropping {} NaN values from lab_events_df_sf".format(
            int(lab_events_df_sf[["valuestr"]].isna().sum().iloc[0])
        )
    )
    lab_events_df_sf = lab_events_df_sf.dropna(subset=["valuestr"])

    print(
        "Dropping {} NaN values from microbiology_df_sf".format(
            int(microbiology_df_sf[["valuestr"]].isna().sum().iloc[0])
        )
    )
    microbiology_df_sf = microbiology_df_sf.dropna(subset=["valuestr"])

    # Drop ___ values
    print(
        "Dropping {} ___ values from lab_events_df_sf".format(
            (lab_events_df_sf["valuestr"] == "___").sum()
        )
    )
    lab_events_df_sf = lab_events_df_sf[lab_events_df_sf["valuestr"] != "___"]

    print(
        "Dropping {} ___ values from microbiology_df_sf".format(
            (microbiology_df_sf["valuestr"] == "___").sum()
        )
    )
    microbiology_df_sf = microbiology_df_sf[microbiology_df_sf["valuestr"] != "___"]

    # Fill in NaN hadm_ids if possible
    (
        lab_events_df_sf,
        radiology_report_df_sf,
        microbiology_df_sf,
    ) = fill_nan_hadm(
        lab_events_df_sf,
        radiology_report_df_sf,
        microbiology_df_sf,
        disease_ids,
        transfers_df,
        hadm_to_subject_id,
    )

    hadm_info = {}

    for _id in disease_ids:
        if _id in discharge_dict:
            discharge_row = discharge_dict[_id]
            discharge_text = discharge_row["text"]

            # Check if history field exists
            if (
                "history of present illness" not in discharge_text.lower()
                and "___ of present illness:" not in discharge_text.lower()
            ):
                continue

            history = extract_history(discharge_text)

            pe = extract_physical_examination(discharge_text)

            le, ref_r_low, ref_r_up = parse_lab_events(lab_events_df_sf, _id)

            microbio, microbio_spec = parse_microbio(microbiology_df_sf, _id)

            rad = extract_rad_events(
                radiology_report_df_sf[radiology_report_df_sf["hadm_id"] == _id][
                    "text"
                ].values
            )

            note_ids = radiology_report_df_sf[radiology_report_df_sf["hadm_id"] == _id][
                "note_id"
            ].values

            note_names = []
            for note_id in note_ids:
                name = exam_name_map.get(note_id, None)
                if name is None:
                    parent_note_id = parent_note_map.get(note_id, None)
                    if parent_note_id:
                        name = exam_name_map.get(parent_note_id, "Unknown")
                    else:
                        warnings.warn(
                            "Note ID {} has no exam name and no parent_note_id".format(
                                note_id
                            )
                        )
                        name = ""
                note_names.append(name)

            rad_regions = []
            rad_modalities = []
            for exam_name in note_names:
                # Count matches of each modality and region and get most frequent plus counts
                (
                    frequent_modality,
                    frequent_modality_count,
                    frequent_region,
                    frequent_region_count,
                ) = count_radiology_modality_and_organ_matches(exam_name)

                if frequent_modality_count == 0:
                    frequent_modality = None
                if frequent_region_count == 0:
                    frequent_region = None

                rad_modalities.append(frequent_modality)
                rad_regions.append(frequent_region)

            rad_data = []
            for i in range(len(rad)):
                rad_data.append(
                    {
                        "Report": rad[i],
                        "Modality": rad_modalities[i],
                        "Region": rad_regions[i],
                        "Exam Name": note_names[i],
                        "Note ID": note_ids[i],
                    }
                )

            hadm_info[_id] = {
                "Discharge": discharge_text,
                "Patient History": history,
                "Physical Examination": pe,
                "Laboratory Tests": le,
                "Microbiology": microbio,
                "Microbiology Spec": microbio_spec,
                "Reference Range Lower": ref_r_low,
                "Reference Range Upper": ref_r_up,
                "Radiology": rad_data,
            }
        else:
            skipped += 1
    print("Skipped {} hadm_ids".format(skipped))
    return hadm_info


# Examine completeness of data
def check_missing(hadm_info, pathology):
    print("There are {} patients.".format(len(hadm_info)))
    missing = 0
    missing_history = 0
    missing_pe = 0
    missing_le = 0
    missing_rad = 0
    bad_dd = 0
    bad_ids = set()
    for _id in hadm_info:
        if (
            _id not in hadm_info
        ):  # Those not in hadm_info are completely missing discharge or history. CORRECT!
            missing += 1
        #  print('Missing {}'.format(_id))
        #  if len(discharge_df[discharge_df['hadm_id']==_id]) != 0:
        #    discharge = discharge_df[discharge_df['hadm_id']==_id]['text'].values[0]
        #    print(discharge)
        else:
            # Check history
            if hadm_info[_id]["Patient History"] == "":
                missing_history += 1
                bad_ids.add(_id)
            # Check pe
            if (
                hadm_info[_id]["Physical Examination"] == ""
                or len(hadm_info[_id]["Physical Examination"]) < 40
            ):  # Small strings are not valid PEs
                # if len(hadm_info[_id]['Physical Examination']) < 40:
                #  print(hadm_info[_id]['Physical Examination'])
                missing_pe += 1
                bad_ids.add(_id)
            # Check le
            if len(hadm_info[_id]["Laboratory Tests"]) == 0:
                missing_le += 1
                bad_ids.add(_id)
            # Check rad
            abdomen_imaged = False
            for rad in hadm_info[_id]["Radiology"]:
                if rad["Region"] == "Abdomen" and rad["Modality"] is not None:
                    abdomen_imaged = True
            if not abdomen_imaged:
                missing_rad += 1
                bad_ids.add(_id)
            # Check diagnosis
            if not pathology_in_primary_diagnosis(
                pathology.lower(), hadm_info[_id]["Discharge Diagnosis"].lower()
            ):
                # print(hadm_info[_id]["Discharge Diagnosis"])
                bad_dd += 1
                bad_ids.add(_id)

    # Remove bad ids to create clean dataset
    hadm_info_clean = hadm_info.copy()
    for _id in bad_ids:
        del hadm_info_clean[_id]

    print("Missing {} patients".format(missing))
    print("Missing {} history".format(missing_history))
    print("Missing {} pe".format(missing_pe))
    print("Missing {} le".format(missing_le))
    print("{} subjects without abdomen rad".format(missing_rad))
    print("Missing pathology in {} discharge diagnoses".format(bad_dd))
    print("{} clean subjects".format(len(hadm_info_clean)))
    return hadm_info_clean


def pathology_in_primary_diagnosis(pathology, discharge_diagnosis):
    # Remove secondary diagnosis and everything after
    secondary_start = None
    for line in discharge_diagnosis.split("\n"):
        if "secondary" in line.lower() and not (
            "secondary to" in line.lower() or "with secondary" in line.lower()
        ):
            secondary_start = line
            break
    if secondary_start:
        discharge_diagnosis = discharge_diagnosis.split(secondary_start)[0]
    if pathology in discharge_diagnosis:
        return True
    return False


def sanitize_hadm_texts(hadm_info, disease_names):
    invalid_visits = 0
    for _id in hadm_info:
        inval = False
        for disease_name in disease_names:
            if not inval:
                # Sanitize history - if history contains disease name, invalidate the visit
                if re.search(
                    re.compile(disease_name, re.IGNORECASE),
                    hadm_info[_id]["Patient History"],
                ):
                    hadm_info[_id]["Patient History"] = ""
                    invalid_visits += 1
                    inval = True
                    continue

                # Sanitize physical examination
                hadm_info[_id]["Physical Examination"] = re.sub(
                    re.compile(disease_name, re.IGNORECASE),
                    "____",
                    hadm_info[_id]["Physical Examination"],
                )

                # Sanitize rads
                for i, rad in enumerate(hadm_info[_id]["Radiology"]):
                    hadm_info[_id]["Radiology"][i]["Report"] = re.sub(
                        re.compile(disease_name, re.IGNORECASE), "____", rad["Report"]
                    )
    print(
        "Invalidated {} visits due to pathology reference in patient history".format(
            invalid_visits
        )
    )
    return hadm_info
