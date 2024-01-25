import re


def parse_report(report):
    # Split the report into lines
    lines = report.strip().split("\n")

    # Initialize the dictionary
    report_dict = {}

    # Check if the first line ends with a colon and if not, add it (typically imaging modality)
    if lines[0].isupper() and lines[0].strip()[-1] != ":":
        lines[0] = lines[0].strip() + ":"

    # Check if theres a line of only capital letters to be included (typically imaging modality)
    # TODO: Sometimes there is report text after this line so should consider strategy to extract
    for i, line in enumerate(lines):
        if line.isupper() and ":" not in line:
            lines[i] = line.strip() + ":"

    # Rejoin the remaining lines and parse the report as before
    report = "\n".join(lines)
    pattern = r"(?m)^([A-Z \t,._-]+):((?:(?!^[A-Z \t,._-]+:).)*)"
    sections = re.findall(pattern, report, re.DOTALL)

    # Add the sections to the dictionary
    for section in sections:
        report_dict[section[0].strip()] = section[1].strip()

    return report_dict


def extract_rad_events(texts):
    bad_rad_fields = [
        "CLINICAL HISTORY",
        "MEDICAL HISTORY",
        "CLINICAL INFORMATION",
        "COMPARISON",
        "COMPARISONS",
        "COMMENT",
        "CONCLUSION",
        "HISTORY",
        "IMPRESSION",
        "CLINICAL INDICATION",
        "INDICATION",
        "OPERATORS",
        "REASON",
        "REFERENCE",
        "DATE",
    ]
    cleaned_texts = []
    for text in texts:
        # Convert report to dictionary of sections. Also recieve special lines to be added to beginning
        sections = parse_report(text)
        text_clean = ""
        info_added = False
        for field in sections:
            # only add field if it does not start with any bad field
            if not any([field.startswith(bad_field) for bad_field in bad_rad_fields]):
                if sections[field]:
                    info_added = True
                text_clean += "{}:\n{}\n\n".format(field, sections[field])
            else:
                # print('Removed: {}'.format(field))
                pass
        # Could be that the only usable string was a headline field without any info. Want to remove these
        if not info_added:
            text_clean = ""
        cleaned_texts.append(text_clean)
    return cleaned_texts


# Extract radiology reports from those that didnt have entries in the radiology df
def extract_section_headers(text):
    # Extract headers which is a lines of words that ends in a colon
    pattern = r"(?<=\n)\s*[\w\s]+(?=:):"
    # Print all headers
    section_headers = re.findall(pattern, text, re.MULTILINE)
    section_headers = [header.rsplit("\n", 1)[-1] for header in section_headers]
    return section_headers


def find_prefix_suffix(headers):
    prefixes = [
        "Name:",
        "Admission Date:",
        "Date of Birth:",
        "Service:",
        "Allergies:",
        "Attending:",
        "Chief Complaint:",
        "Major Surgical or Invasive Procedure:",
        "History of Present Illness:",
        "Past Medical History:",
        "Social History:",
        "Family History:",
        "Physical Exam:",
        "Discharge Exam:",
        "Admission Physical Exam:",
        "General:",
        "Pertinent Results:",
        "INDICATIONS FOR CONSULT:",
        "LABORATORY TESTING:",
        "OPERATIVE REPORT:",
        "PREOPERATIVE DIAGNOSIS:",
        "POSTOPERATIVE DIAGNOSIS:",
        "PROCEDURE:",
        "ASSISTANT:",
        "ANESTHESIA:",
        "ESTIMATED BLOOD LOSS:",
        "clinical history",
        "microbiology",
    ]
    suffixes = [
        "IMPRESSION:",
        "Brief Hospital Course:",
        "CONCISE SUMMARY OF HOSPITAL COURSE:",
        "Medications on Admission:",
        "Discharge Medications:",
        "Tablet Refills:",
        "Discharge Disposition:",
        "Discharge Diagnosis:",
        "Discharge Condition:",
        "Mental Status:",
        "Level of Consciousness:",
        "Activity Status:",
        "Discharge Instructions:",
        "the insturctions below regarding your discharge:",
        "Emergency Department for any of the following:",
        "Incision Care:",
        "Followup Instructions:",
    ]

    prefixes = [prefix.lower() for prefix in prefixes]
    suffixes = [suffix.lower() for suffix in suffixes]

    # Using list comprehension to find common elements in headers and prefixes
    prefix_intersection = [header for header in headers if header.lower() in prefixes]
    # Using list comprehension to find common elements in headers and suffixes
    suffix_intersection = [header for header in headers if header.lower() in suffixes]

    # If there are common elements, return the last element from prefixes and the first from suffixes
    if prefix_intersection and suffix_intersection:
        return prefix_intersection[-1], suffix_intersection[0]
    else:
        return None, None


def sanitize_rad(hadm_info):
    # Remove rad reports if no modality or region is found or report is empty
    removed_cnt = 0
    subj_removed_cnt = 0
    for _id in list(hadm_info.keys()):
        to_del = []
        for i, rad in enumerate(hadm_info[_id]["Radiology"]):
            if rad["Modality"] is None or rad["Region"] is None or rad["Report"] == "":
                to_del.append(i)
        for i in sorted(to_del, reverse=True):
            del hadm_info[_id]["Radiology"][i]
        removed_cnt += len(to_del)
        if len(to_del) > 0:
            subj_removed_cnt += 1
    print(
        "Removed {} rad reports from {} subjects".format(removed_cnt, subj_removed_cnt)
    )
    return hadm_info
