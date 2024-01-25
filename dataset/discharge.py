import re

from dataset.utils import regex_extracter, last_substring_index


def extract_chief_complaints(hadm_ids, discharge_df):
    """
    Loops over patients and extracts chief complaints from discharge summaries. Extracts from chief complaint field to major surgical field.

    Args:
        hadm_ids (list): List of hadm_ids of patients to extract chief complaints from
        discharge_df (pd.DataFrame): DataFrame containing discharge summaries

    Returns:
        ccs (list): List of chief complaints
        cc_ids (list): List of hadm_ids with valid chief complaints
        discharge_cntr (int): Number of valid discharge summaries that were looped over (i.e. without empty discharge that were skipped)
    """
    ccs = []
    discharge_cntr = 0
    cc_ids = []
    for _id in hadm_ids:
        if len(discharge_df[discharge_df["hadm_id"] == _id]) == 0:
            continue
        discharge_cntr += 1
        discharge = discharge_df[discharge_df["hadm_id"] == _id]["text"].values[0]
        cc = extract_cc(discharge)
        if len(cc) > 0:
            ccs.append(cc[0].strip())
            cc_ids.append(_id)
    return ccs, cc_ids, discharge_cntr


def extract_cc(text):
    regex = re.compile(
        "(?:chief|___) complaint:(.*)major (?:surgical|___)",
        re.IGNORECASE | re.DOTALL,
    )
    cc = regex.findall(text)
    return cc


def extract_history(text):
    """
    Extract initial complaint (Patient History) from discharge summary text. Extract from 'history of present illness:' field to 'physical exam' field using regex. Case insensitive.

    Args:
        text (str): Discharge summary text

    Returns:
        text (str): Extracted patient history
    """
    #
    text = text.replace("\n", " ")

    success = False
    i = 0
    pe_strings = [
        "physical exam:",
        "physical examination:",
        "physical ___:",
        "pe:",
        "pe ___:",
        "(?:pertinent|___) results:",
        "hospital course:",
    ]
    while not success and i < len(pe_strings):
        regex = re.compile(
            f"(?:history|___) of present(?:ing)? illness:.*?{pe_strings[i]}",
            re.IGNORECASE | re.DOTALL,
        )
        text, success = regex_extracter(text, regex)
        i += 1
    if not success:
        print(text)
        return ""
        # raise Warning("No history match found")

    # remove header
    text = re.sub(
        re.compile("history of present(?:ing)? illness:", re.IGNORECASE), "", text
    )

    # remove terminal string
    for pe_str in pe_strings:
        text = re.sub(re.compile(pe_str, re.IGNORECASE), "", text)

    return text


def extract_diagnosis_from_discharge(text):
    start_headers = ["discharge diagnosis:", "___ diagnosis:"]
    end_headers = [
        "discharge condition:",
        "___ condition:",
        "condition:",
        "procedure:",
        "procedures:",
        "invasive procedure on this admission:",
    ]
    start = 0
    for start_header in start_headers:
        if start_header in text.lower():
            pos_start = last_substring_index(text.lower(), start_header)
            if pos_start != -1:
                start = max(start, pos_start + len(start_header))
    if not start:
        # As last resort match against empty string which sometimes has diagnosis for some reason
        start_header = "\n___:"
        if start_header in text.lower():
            pos_start = last_substring_index(text.lower(), start_header)
            if pos_start != -1:
                start = pos_start
        else:
            raise Exception("No start header found")
    end = 0
    for end_header in end_headers:
        if end_header in text.lower():
            pos_end = last_substring_index(text.lower(), end_header)
            if pos_end != -1:
                end = max(end, pos_end)
            break
    if not end:
        raise Exception("No end header found")
    discharge_diagnosis = text[start:end]
    return discharge_diagnosis.strip()


def extract_physical_examination(text):
    # extract from 'physical exam:' to 'pertinent results:' using regex. Case insensitive
    text = text.replace("\n", " ")
    success = False
    i = 0
    pe_strings = [
        "physical exam:",
        "physical examination:",
        "physical ___:",
        "pe:",
        "pe ___:",
        "pertinent results:",
    ]
    while not success and i < len(pe_strings):
        terminal_str = "pertinent results:"
        if terminal_str not in text.lower():
            terminal_str = "brief hospital course:"
        regex = re.compile(
            f"{pe_strings[i]}.*?{terminal_str}", re.IGNORECASE | re.DOTALL
        )
        text, success = regex_extracter(text, regex)
        i += 1
    if not success:
        return ""

    # remove header
    for pe_str in pe_strings:
        text = re.sub(re.compile(pe_str, re.IGNORECASE), "", text)

    # remove terminal string
    text = re.sub(re.compile("pertinent results:", re.IGNORECASE), "", text)
    text = re.sub(re.compile("brief hospital course:", re.IGNORECASE), "", text)

    # remove everything after discharge pe
    text = re.sub(re.compile("at discharge.*", re.IGNORECASE), "", text)
    text = re.sub(re.compile("upon discharge.*", re.IGNORECASE), "", text)
    text = re.sub(re.compile("on discharge.*", re.IGNORECASE), "", text)
    text = re.sub(re.compile("discharge.*", re.IGNORECASE), "", text)

    return text
