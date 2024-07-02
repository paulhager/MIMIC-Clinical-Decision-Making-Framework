from os.path import join
import pickle
import re


def regex_extracter(text, regex):
    """
    Extract text using regex. If no match, return original text.

    Args:
        text (str): Text to extract from
        regex (str): Regex to use for extraction

    Returns:
        text (str): Extracted text which matches entire regex or original text if no match
        success (bool): True if match found, False otherwise
    """
    try:
        return re.search(regex, text).group(0), True
    except Exception:
        return text, False


def last_substring_index(larger_string, substring):
    last_index = -1
    while True:
        index = larger_string.find(substring, last_index + 1)
        if index == -1:
            break
        last_index = index
    return last_index


# Write fields as csv with $ separator
def write_hadm_to_file(hadm_info, filename, base_mimic=""):
    # Write pickle for easy loading
    with open(join(base_mimic, filename + ".pkl"), "wb") as f:
        pickle.dump(hadm_info, f)


# Load from pickle
def load_hadm_from_file(filename, base_mimic=""):
    with open(join(base_mimic, filename + ".pkl"), "rb") as f:
        hadm_info = pickle.load(f)
    return hadm_info


# Create a function to nicely print the results
def print_value_counts(value_counts, n):
    print(f"{'Value':<100} | Count")
    print("----------------------")
    for index, value in value_counts.head(n).items():
        print(f"{index:<100} | {value}")
