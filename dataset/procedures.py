import re
from icd.procedure_mappings import icd_converter, uniqueify_lists


def extract_procedure_from_discharge_summary(discharge_summary):
    # Extracts everything after the "Major Surgical or Invasive Procedure:" line until the next empty line
    # Returns a list of procedures
    procedure_substrings = [
        "Major Surgical or Invasive Procedure:",
        "PROCEDURES:",
        "PROCEDURE:",
        "Major Surgical ___ Invasive Procedure:",
        "___ Surgical or Invasive Procedure:",
        "INVASIVE PROCEDURE ON THIS ADMISSION:",
        "Major ___ or Invasive Procedure:",
        "MAJOR SURGICAL AND INVASIVE PROCEDURES PERFORMED THIS DURING\nADMISSION:",
    ]
    for substring in procedure_substrings:
        pattern = rf"{re.escape(substring)}.*?\n\s*\n"
        match = re.search(pattern, discharge_summary, re.DOTALL)
        if match:
            procedures_string = match.group(0)

            # Remove section title
            procedures_string = procedures_string.replace(substring, "")

            # Replace newline with space to make one sentence
            procedures_string = procedures_string.replace("\n", " ")

            # Split on delimiters
            procedures = re.split(r"\: |, |\. | - ", procedures_string)

            # Clean
            procedures = [proc.strip() for proc in procedures if proc.strip() != ""]
            return procedures
    return []


def extract_procedures(hadm_info, procedures_df_icd9, procedures_df_icd10):
    for _id in hadm_info:
        discharge_procedures = extract_procedure_from_discharge_summary(
            hadm_info[_id]["Discharge"]
        )
        if len(discharge_procedures) == 0:
            print("No procedures found for {}".format(_id))
        hadm_info[_id]["Procedures Discharge"] = discharge_procedures

        procedures_icd9 = procedures_df_icd9[procedures_df_icd9["hadm_id"] == _id][
            "icd_code"
        ].values
        hadm_info[_id]["Procedures ICD9"] = procedures_icd9.tolist()
        hadm_info[_id]["Procedures ICD9"] = [int(p) for p in procedures_icd9]

        procedures_str = procedures_df_icd9[procedures_df_icd9["hadm_id"] == _id][
            "long_title"
        ].values
        hadm_info[_id]["Procedures ICD9 Title"] = procedures_str.tolist()

        procedures_icd10 = procedures_df_icd10[procedures_df_icd10["hadm_id"] == _id][
            "icd_code"
        ].values
        hadm_info[_id]["Procedures ICD10"] = procedures_icd10.tolist()
        hadm_info[_id]["Procedures ICD10"] = [str(p) for p in procedures_icd10]

        procedures_str = procedures_df_icd10[procedures_df_icd10["hadm_id"] == _id][
            "long_title"
        ].values
        hadm_info[_id]["Procedures ICD10 Title"] = procedures_str.tolist()
    return hadm_info


def generate_colectomy_procedures(diag_icd, procedures_df):
    # To generate possible procedures for colectomy examine the procedures done on those patients with diverticulitis with perforation
    perforation_hadm_df = diag_icd[
        diag_icd["long_title"].str.contains(
            "diverticulitis.*with perforation", case=False
        )
    ][["hadm_id"]].drop_duplicates()
    print(
        "{} subjects with diverticulitis with perforation".format(
            len(perforation_hadm_df)
        )
    )
    perforation_procedures_df = perforation_hadm_df.merge(
        procedures_df, on="hadm_id", how="left"
    )
    unique_procedures = list(perforation_procedures_df.long_title.unique())
    unique_procedures = [o for o in unique_procedures if o == o]
    unique_procedures.sort()
    # This list seems to be populated only by ICD10 codes for some reason
    # for o in unique_procedures:
    #  print(o)

    # This list was then given to a medical expert to select the relevant procedures resulting in the following list
    colectomy_procedures_long_title_str_prefixes = [
        "Excision of Cecum",
        "Resection of Cecum",
        "Excision of Descending Colon",
        "Resection of Descending Colon",
        "Excision of Large Intestine",
        "Release Large Intestine",
        "Resection of Right Large Intestine",
        "Excision of Left Large Intestine",
        "Excision of Small Intestine",
        "Release Small Intestine" "Excision of Sigmoid Colon",
        "Release Sigmoid Colon",
        "Resection of Sigmoid Colon",
        "Release Transverse Colon",
        "Resection of Transverse Colon",
        "Resection of Ascending Colon",
        "Excision of Duodenum",
        "Release Duodenum",
        "Excision of Ileum",
        "Release Ileum",
        "Excision of Jejunum",
        "Release Jejunum",
    ]

    # Get the ICD10 codes of each colectomy procedure and save to a dictionary. The procedure strings are prefixes of the long_titles so we need to check for starts with and not exact equals

    # Filtering procedures dataframe to only contain rows with icd_version == 10
    proc_df = procedures_df[procedures_df["icd_version"] == 10]

    # Use str.startswith() method to match any of the prefixes. Don't include those with diagnostic in the name
    mask = proc_df["long_title"].apply(
        lambda x: any(
            x.startswith(prefix) and "Diagnostic" not in x
            for prefix in colectomy_procedures_long_title_str_prefixes
        )
    )

    # Filter the dataframe to only contain rows that match the prefixes
    filtered_df = proc_df[mask]

    # Convert the filtered dataframe to a dictionary where 'long_title' is the key and 'icd_code' is the value
    colectomy_procedure_mapping = filtered_df.set_index("long_title")[
        "icd_code"
    ].to_dict()

    print("'" + "', '".join(colectomy_procedure_mapping.keys()) + "'")
    print("'" + "', '".join(colectomy_procedure_mapping.values()) + "'")

    # https://www.cms.gov/medicare/coding/icd9providerdiagnosticcodes/codes
    # https://www.cms.gov/medicare/coding/icd9providerdiagnosticcodes/downloads/icd-9-cm-v32-master-descriptions.zip
    procedure_names_icd9_path = "./icd/CMS32_DESC_LONG_SG.txt"

    # https://www.cms.gov/files/zip/2024-icd-10-pcs-codes-file.zip
    procedure_names_icd10_path = "./icd/icd10pcs_codes_2024.txt"

    # https://www.cms.gov/medicare/coding/icd10/2018-icd-10-pcs-and-gems
    # https://www.cms.gov/medicare/coding/icd10/downloads/2018-icd-10-pcs-general-equivalence-mappings.zip
    icd9_to_10_mapping_path = "./icd/gem_i9pcs.txt"
    icd10_to_9_mapping_path = "./icd/gem_pcsi9.txt"

    icd_9_codes, icd_9_titles = icd_converter(
        colectomy_procedure_mapping.values(),
        10,
        procedure_names_icd9_path,
        procedure_names_icd10_path,
        icd9_to_10_mapping_path,
        icd10_to_9_mapping_path,
    )
    icd_9_codes_u, icd_9_titles_u = uniqueify_lists(icd_9_codes, icd_9_titles)

    print("'" + "', '".join(icd_9_codes_u) + "'")
    print("'" + "', '".join(icd_9_titles_u) + "'")
