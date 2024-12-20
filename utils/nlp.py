import logging
from typing import List
import string
import copy

import pandas as pd
import spacy
from negspacy.negation import Negex  # noqa: F401
import nltk
import re
from thefuzz import process, fuzz
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from transformers import LlamaTokenizer, AutoTokenizer
from exllamav2 import ExLlamaV2Tokenizer
import tiktoken

from tools.utils import FLUID_MAPPING, itemid_to_field

nlp = spacy.load("en_core_sci_lg")
nlp.add_pipe(
    "negex",
    config={
        "chunk_prefix": ["no"],
    },
    last=True,
)
# nltk.download("stopwords")

###
# Collection of functions for natural language processing utility
###


def treatment_alternative_procedure_checker(operation_keywords, text):
    for alternative_operations in operation_keywords:
        op_loc = alternative_operations["location"]
        for op_mod in alternative_operations["modifiers"]:
            for sentence in text.split("."):
                if keyword_positive(sentence, op_loc) and keyword_positive(
                    sentence, op_mod
                ):
                    return True
    return False


# Makes check if a keyword is positive i.e. occurs and is not negated. For negation check uses the negex algorithm i.e. "No appendicitis" or "No signs of appendicitis" or "Abscence of typical indications of appendicitis"
def keyword_positive(sentence, keyword):
    doc = nlp(sentence)

    for e in doc.ents:
        if keyword.lower() in e.text.lower():
            return not e._.negex

    # Just check for keyword in sentence if not found in entities
    return keyword.lower() in sentence.lower()
    # return False


def remove_punctuation(input_string):
    # Make a translator object that will replace punctuation with None (which removes it)
    translator = str.maketrans("", "", string.punctuation)
    return input_string.translate(translator)


def contains(keyword: str, strings: List[str]):
    return any(keyword_positive(string, keyword) for string in strings)


# Check if diagnosis is in list of diagnoses. Combines discharge text diagnosis with all recorded ICD diagnoses
def diagnosis_checker(discharge_diagnosis: str, icd_diagnoses: List[str], keyword: str):
    diags = copy.deepcopy(icd_diagnoses)
    diags.append(discharge_diagnosis)
    return contains(keyword, diags)


def procedure_checker(
    valid_procedures: List,
    done_procedures: List,
):
    for valid_procedure in valid_procedures:
        if type(valid_procedure) == int:
            if valid_procedure in done_procedures:
                return True
        else:
            for done_procedure in done_procedures:
                if keyword_positive(done_procedure, valid_procedure):
                    return True


# Extract keywords from text using spacy library. Keywords are nouns and adjectives
def extract_keywords_spacy(text: str):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    keywords = [token.text for token in doc if token.pos_ in ["NOUN", "ADJ", "PROPN"]]
    return keywords


# Extract keywords from text using nltk library. Keywords are nouns and adjectives
def extract_keywords_nltk(text: str):
    words = word_tokenize(text)
    pos_tags = nltk.pos_tag(words)
    keywords = [word for word, tag in pos_tags if tag in ["NN", "NNS", "JJ", "NNP"]]
    return keywords


# Abbreviations (i.e. short names) are often given in parantheses after long name. Extract both short and long name
def extract_short_and_long_name(test: str):
    match = re.search("\((.*)\)", test)
    if match:
        short_name = match.group(1).strip()
        long_name = test.replace(match.group(0), "").strip()
        long_name = long_name.replace("  ", " ")
        long_name = long_name.replace(" ,", ",")
        return short_name, long_name
    else:
        return test, test


# Check if a fluid was specifically requested. Used for itemid filtering
def match_fluid(test: str):
    for fluid, keywords in FLUID_MAPPING.items():
        for keyword in keywords:
            if keyword.lower() in test.lower():
                return fluid, re.sub(keyword, "", test, flags=re.IGNORECASE).strip()
    return None, None


# Convert list of tests to canonical names. Canonical names are the names used in the lab test mapping file
def convert_labs_to_itemid(tests: List[str], lab_test_mapping_df: pd.DataFrame):
    logging.getLogger().setLevel(logging.ERROR)
    labels = lab_test_mapping_df["label"].tolist()
    all_tests = []
    for test_full in tests:
        fluid, test_no_fluid = match_fluid(test_full)

        # Extract short and long name
        test_short, test_long = extract_short_and_long_name(test_full)

        # Try fuzzy matching to allow for spelling mistakes and small discrepencies. Use ratio because we have many tests that are just one letter that match too strong with partial ratio
        # Start with full name since its hardest to match and has least amount of false positives
        test_match, score = process.extractOne(test_full, labels, scorer=fuzz.ratio)
        if score < 90:
            # If no match, try using the long name.
            test_match, score = process.extractOne(test_long, labels, scorer=fuzz.ratio)
            if score < 90:
                # If no match, try using the short name but look for exact match because a single letter difference typically completely changes the test
                test_match, score = process.extractOne(
                    test_short, labels, scorer=fuzz.ratio
                )
                if score < 100:
                    # If no match, try removing the fluid and searching again
                    if fluid:
                        labels_fluid = lab_test_mapping_df[
                            lab_test_mapping_df["fluid"] == fluid
                        ]["label"].tolist()
                        test_match, score = process.extractOne(
                            test_no_fluid, labels_fluid, scorer=fuzz.ratio
                        )
                        if score < 90:
                            # Finally, return just test itself. Will not match going forward but saves original intent
                            test_match = ""
                            with open("no_canonical_names.txt", "a") as f:
                                f.write(f"{test_full}\n")
                    else:
                        # Finally, return just test itself. Will not match going forward but saves original intent
                        test_match = ""
                        with open("no_canonical_names.txt", "a") as f:
                            f.write(f"{test_full}\n")

        # Replace test with full list of valid names if matched
        if test_match:
            expanded_tests = lab_test_mapping_df.loc[
                lab_test_mapping_df["label"] == test_match, "corresponding_ids"
            ].iloc[0]

            # Only include those of specific fluid if specified
            if fluid:
                expanded_tests = [
                    test
                    for test in expanded_tests
                    if (itemid_to_field(test, "fluid", lab_test_mapping_df) == fluid)
                    or (  # If fluid is nan then its a microbio test. TODO: Check against spec_itemid instead
                        itemid_to_field(test, "fluid", lab_test_mapping_df)
                        != itemid_to_field(test, "fluid", lab_test_mapping_df)
                    )
                ]
            all_tests.extend(expanded_tests)
        else:
            all_tests.append(test_full)
    logging.getLogger().setLevel(logging.INFO)
    return all_tests


def remove_stop_words(sentence):
    nltk_stop_words = set(stopwords.words("english"))

    # Keep uppercase single letters as they often are part of lab tests
    lowercase_single_letters = {w for w in nltk_stop_words if len(w) == 1}
    nltk_stop_words = nltk_stop_words - lowercase_single_letters

    # Split words and keep punctuation
    words_with_punct = sentence.split(" ")
    words_without_punct = [remove_special_characters(w) for w in words_with_punct]

    # Remove stop words
    filtered_sentence = " ".join(
        word
        for i, word in enumerate(words_with_punct)
        if words_without_punct[i].lower() not in nltk_stop_words
        and words_without_punct[i] not in lowercase_single_letters
    )

    # Remove A if it is the first word
    filtered_sentence = re.sub(r"\bA\s", "", filtered_sentence)

    return filtered_sentence


def remove_special_characters(word):
    return re.sub(r"[^\w\s]", "", word)


def extract_sections(text, tags_list):
    tags = {
        "system": (tags_list["system_tag_start"], tags_list["system_tag_end"]),
        "user": (tags_list["user_tag_start"], tags_list["user_tag_end"]),
        "assistant": (tags_list["ai_tag_start"], tags_list["ai_tag_end"]),
    }

    pattern = "|".join(
        [
            f"(?P<{tag}>{re.escape(start)}(.*?){re.escape(end)})"
            for tag, (start, end) in tags.items()
        ]
    )

    sections = []
    last_end = 0
    for match in re.finditer(pattern, text, re.DOTALL):
        for tag, content in match.groupdict().items():
            if content is not None:
                sections.append(
                    {
                        "role": tag,
                        "content": content.replace(tags[tag][0], "")
                        .replace(tags[tag][1], "")
                        .strip(),
                    }
                )
                last_end = match.end()

    # Check if we have an open tag without close
    remaining_text = text[last_end:].strip()
    for tag, (start, end) in tags.items():
        if remaining_text.startswith(start):
            sections.append(
                {"role": tag, "content": remaining_text[len(start) :].strip()}
            )

    return sections


# Text parses differently if done line by line and as a whole. This function extracts the first diagnosis from the text by checking both
def extract_primary_diagnosis(text):
    earliest_keyword_index = len(text)

    # Do parsing of entire text and check for earliest possible diagnosis
    doc = nlp(text)
    diag = check_ents_for_diagnosis_noun_chunks(doc)
    if diag:
        earliest_keyword_index = min(earliest_keyword_index, text.find(diag))
    diag = check_ents_for_diagnosis_entities(doc)
    if diag:
        earliest_keyword_index = min(earliest_keyword_index, text.find(diag))

    # Do parsing of each line and check for earliest possible diagnosis
    for line in text.split("\n"):
        doc = nlp(line)
        diag = check_ents_for_diagnosis_noun_chunks(doc)
        if diag:
            earliest_keyword_index = min(earliest_keyword_index, text.find(diag))
            break
        diag = check_ents_for_diagnosis_entities(doc)
        if diag:
            earliest_keyword_index = min(earliest_keyword_index, text.find(diag))
            break

    # Extract line with diagnosis and make sure we only return the first diagnosis if multiple given on one line
    prim_diag = text[earliest_keyword_index:]
    prim_diag = prim_diag.split("\n")[0]
    prim_diag = prim_diag.split(",")[0]
    # We can split on 'and' here because we are just looking for the general pathology, not a specific subtype i.e. "diverticulitis with perforation and stricture and abscess"
    prim_diag = re.split(r"\band\b", prim_diag)[0]
    prim_diag = re.split(r"\bor\b", prim_diag)[0]
    # prim_diag = re.split(r"\bwith\b", prim_diag)[0]
    prim_diag = re.split(r"\bvs[.]?\b", prim_diag)[0]
    # prim_diag = re.split(r"\/", prim_diag)[0]
    return prim_diag.strip()


def check_ents_for_diagnosis_entities(doc):
    for e in doc.ents:
        d = e.text.lower()
        if (
            "primary" not in d
            and "diagnosis" not in d
            and "diagnoses" not in d
            and "dx" not in d
            and d != "active"
            and d != "acute"
        ):
            return e.text
    return None


def check_ents_for_diagnosis_noun_chunks(doc):
    for chunk in doc.noun_chunks:
        d = chunk.text.lower()
        # Remove initial characters to first letter
        d = re.sub(r"^[^a-zA-Z]+", "", d)
        if (
            "primary" not in d
            and "diagnosis" not in d
            and "diagnoses" not in d
            and "dx" not in d
            and d != "active"
            and d != "acute"
        ):
            return d.strip()
    return None


def calculate_num_tokens(tokenizer, inputs):
    num_tokens = 0
    for input in inputs:
        tokens = tokenizer.encode(input)
        if isinstance(tokenizer, ExLlamaV2Tokenizer):
            num_tokens += tokens.shape[-1]
        else:
            num_tokens += len(tokens)
    return num_tokens


def truncate_text(tokenizer, input, available_tokens):
    if isinstance(tokenizer, ExLlamaV2Tokenizer):
        truncated_input_tokens = tokenizer.encode(input)[:, :available_tokens]
        input = tokenizer.decode(truncated_input_tokens)[0]
    elif isinstance(tokenizer, tiktoken.Encoding):
        truncated_input_tokens = input = tokenizer.encode(input)[:available_tokens]
        input = tokenizer.decode(truncated_input_tokens)
    else:
        truncated_tokens = tokenizer.encode(
            input,
            truncation=False,
            padding=False,
        )[:available_tokens]
        input = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
    return input


def create_lab_test_string(
    test_id,
    lab_test_mapping_df,
    hadm_info,
    include_ref_range=False,
    bin_lab_results=False,
    bin_lab_results_abnormal=False,
    only_abnormal_labs=False,
):
    lab_test_fluid = itemid_to_field(test_id, "fluid", lab_test_mapping_df)
    lab_test_label = itemid_to_field(test_id, "label", lab_test_mapping_df)
    lab_test_value = hadm_info["Laboratory Tests"].get(test_id, "N/A")

    # If test not found in lab tests, check microbiology
    if lab_test_value == "N/A":
        lab_test_fluid = "Microbiology"
        lab_test_value = hadm_info["Microbiology"].get(test_id, "N/A")

    if only_abnormal_labs:
        rr_lower = hadm_info["Reference Range Lower"].get(test_id, None)
        rr_upper = hadm_info["Reference Range Upper"].get(test_id, None)
        if rr_lower == rr_lower and rr_upper == rr_upper:
            try:
                lab_test_value_split = float(lab_test_value.split()[0])
                if (lab_test_value_split > rr_lower) and (
                    lab_test_value_split < rr_upper
                ):
                    return ""
            except ValueError:
                pass

    if bin_lab_results_abnormal:
        if include_ref_range:
            raise ValueError(
                "Binning and printing ref range concurrently not supported"
            )
        rr_lower = hadm_info["Reference Range Lower"].get(test_id, None)
        rr_upper = hadm_info["Reference Range Upper"].get(test_id, None)
        if rr_lower == rr_lower and rr_upper == rr_upper:
            try:
                lab_test_value_split = float(lab_test_value.split()[0])
                if lab_test_value_split < rr_lower or lab_test_value_split > rr_upper:
                    lab_test_value = "Abnormal"
                else:
                    lab_test_value = "Normal"
            except ValueError:
                pass

    if bin_lab_results:
        if include_ref_range:
            raise ValueError(
                "Binning and printing ref range concurrently not supported"
            )
        rr_lower = hadm_info["Reference Range Lower"].get(test_id, None)
        rr_upper = hadm_info["Reference Range Upper"].get(test_id, None)
        if rr_lower == rr_lower and rr_upper == rr_upper:
            try:
                lab_test_value_split = float(lab_test_value.split()[0])
                if lab_test_value_split < rr_lower:
                    lab_test_value = "Low"
                elif lab_test_value_split > rr_upper:
                    lab_test_value = "High"
                else:
                    lab_test_value = "Normal"
            except ValueError:
                pass

    lab_test_str = f"({lab_test_fluid}) {lab_test_label}: {lab_test_value}"

    if include_ref_range:
        rr_lower = hadm_info["Reference Range Lower"].get(test_id, None)
        rr_upper = hadm_info["Reference Range Upper"].get(test_id, None)
        if rr_lower == rr_lower and rr_upper == rr_upper:
            lab_test_str += f" | RR: [{rr_lower} - {rr_upper}]"

    lab_test_str += "\n"
    return lab_test_str


def latex_escape(text):
    """
    :param text: a plain text message
    :return: the message escaped to appear correctly in LaTeX
    """
    conv = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\^{}",
        "\\": r"\textbackslash{}",
        "<": r"\textless{}",
        ">": r"\textgreater{}",
    }
    regex = re.compile(
        "|".join(
            re.escape(str(key))
            for key in sorted(conv.keys(), key=lambda item: -len(item))
        )
    )
    return regex.sub(lambda match: conv[match.group()], text)
