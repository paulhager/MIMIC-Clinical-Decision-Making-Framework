from enum import Enum
from typing import Union, List, Dict

import pandas as pd
from thefuzz import process

from utils.nlp import create_lab_test_string
from tools.utils import UNIQUE_TO_BROAD_MODALITY, itemid_to_field
from agents.prompts import (
    DIAGNOSTIC_CRITERIA_APPENDICITIS,
    DIAGNOSTIC_CRITERIA_CHOLECYSTITIS,
    DIAGNOSTIC_CRITERIA_DIVERTICULITIS,
    DIAGNOSTIC_CRITERIA_PANCREATITIS,
)


class Actions(Enum):
    Physical_Examination = "Physical Examination"
    Laboratory_Tests = "Laboratory Tests"
    Imaging = "Imaging"
    Diagnostic_Criteria = "Diagnostic Criteria"
    Final_Diagnosis = "Final Diagnosis"


def is_valid_action(action: str) -> bool:
    return action in [action.value for action in Actions]


def get_action_results(
    action: str,
    action_results: Dict = None,
    action_input: Union[List[Union[int, str]], Dict] = None,
    lab_test_mapping_df: pd.DataFrame = None,
    include_ref_range: bool = False,
    bin_lab_results: bool = False,
    already_requested_scans: Dict = None,
):
    # Write header
    result_string = f"{action.value}:\n"

    # Lab tests have many names and abbreviations, so we need to check for all of them
    if action == Actions.Laboratory_Tests:
        result_string += retrieve_lab_tests(
            action_input=action_input,
            action_results=action_results,
            lab_test_mapping_df=lab_test_mapping_df,
            include_ref_range=include_ref_range,
            bin_lab_results=bin_lab_results,
        )

    # Imaging tests are made up of a modality and a region
    elif action == Actions.Imaging:
        result_string += retrieve_imaging(
            action_input=action_input,
            action_results=action_results,
            already_requested_scans=already_requested_scans,
        )

    # Simple dictionary lookup
    elif action == Actions.Physical_Examination:
        result_string += retrieve_physical_examination(action_results=action_results)

    # Simple dictionary lookup
    elif action == Actions.Diagnostic_Criteria:
        result_string += retrieve_diagnostic_criteria(
            action_input=action_input,
        )

    # Should never reach here
    else:
        raise ValueError(
            "The only valid actions are Physical Examination, Laboratory Tests, Imaging, and Diagnostic Criteria. Received: {}".format(
                action.value
            )
        )

    return result_string


def retrieve_physical_examination(action_results: Dict) -> str:
    """Returns the results of the physical examination.

    Args:
        action_results (Dict): The results of the physical examination.

    Returns:
        result_string (str): The results of the physical examination in pretty string format to be given as an observation to the model.
    """
    # Retrieve results of physical examination
    action_result = action_results.get("Physical Examination", "Not available.")
    return f"{action_result}\n"


def retrieve_lab_tests(
    action_input: List[Union[int, str]],
    action_results: Dict,
    lab_test_mapping_df: pd.DataFrame,
    include_ref_range: bool,
    bin_lab_results: bool,
) -> str:
    """Retrieves the desired itemids from the patient records

    Args:
        action_input (Union[List[str], Dict]): The requested laboratory tests.
        action_results (Dict): Contains the results of the laboratory tests.
        lab_test_mapping_path (str): The path to the lab test mapping.

    Returns:
        result_string (str): The results of the requested laboratory tests in pretty string format to be given as an observation to the model.
    """
    result_string = ""
    # Action input has already been converted to itemids and expanded during parsing.
    for test in action_input:
        # Only includes tests that are found.
        if test in action_results["Laboratory Tests"]:
            result_string += create_lab_test_string(
                test,
                lab_test_mapping_df,
                action_results,
                include_ref_range,
                bin_lab_results,
            )
        # else:
        #    result_string += (
        #        f"{itemid_to_field(test, 'label', lab_test_mapping_df)}: N/A\n"
        #    )

        # Those tests that were requested but not able to be matched to anything in the dataset are in action_input as strings.
        # TODO: Expand to also include those tests that were requested and found but not in the patient's records.
        if isinstance(test, str):
            result_string += f"{test}: N/A\n"

    return result_string


def retrieve_imaging(
    action_input: Union[List[str], Dict],
    action_results: Dict,
    already_requested_scans: Dict,
) -> str:
    """Finds appropriate scan from list of scans taken and returns. Returns scans in chronological order over multiple requests if multiple present

    Args:
        action_input (Union[List[str], Dict]): The requested imaging scan.
        action_results (Dict): Contains the results of the imaging scans.

    Returns:
        result_string (str): The results of the requested imaging scan in pretty string format to be given as an observation to the model.

    """
    result_string = ""
    # Because the inputs have already been parsed, only defined regions and modalities should be present
    requested_scan = f"{action_input['region']} {action_input['modality']}"
    repeat_scan_index = already_requested_scans.get(requested_scan, 0)
    result = None
    for rad in action_results["Radiology"]:
        if (
            rad["Modality"] == action_input["modality"]
            or UNIQUE_TO_BROAD_MODALITY.get(rad["Modality"], None)
            == action_input["modality"]
        ) and rad["Region"] == action_input["region"]:
            if repeat_scan_index == 0:
                result_string += f"{requested_scan}: {rad['Report']}\n"
                if requested_scan not in already_requested_scans:
                    already_requested_scans[requested_scan] = 1
                else:
                    already_requested_scans[requested_scan] += 1
                return result_string
            else:
                result = (
                    "Cannot repeat this scan anymore. Try a different imaging modality."
                )
                repeat_scan_index -= 1
    if not result:
        result = "Not available. Try a different imaging modality."
    result_string += f"{requested_scan}: {result}\n"
    return result_string


def retrieve_diagnostic_criteria(
    action_input: Union[List[str], Dict],
) -> str:
    """Returns diagnostic criteria of the specified pathology.

    Args:
        action_input (Union[List[str], Dict]): The requested pathology.

    Returns:
        result_string (str): The diagnostic criteria to be given as an observation to the model.
    """
    result_string = ""
    name_to_criteria = {
        "appendicitis": DIAGNOSTIC_CRITERIA_APPENDICITIS,
        "cholecystitis": DIAGNOSTIC_CRITERIA_CHOLECYSTITIS,
        "diverticulitis": DIAGNOSTIC_CRITERIA_DIVERTICULITIS,
        "pancreatitis": DIAGNOSTIC_CRITERIA_PANCREATITIS,
    }
    for patho in action_input:
        patho_match, score = process.extractOne(patho, name_to_criteria.keys())
        if score >= 80:
            patho = patho_match
        diagnostic_criteria = name_to_criteria.get(patho, None)
        if not diagnostic_criteria:
            result_string += f"Diagnostic criteria for {patho} is not available.\n"
            with open("no_diagnostic_criteria.txt", "a") as f:
                f.write(f"{patho}\n")
        else:
            result_string += f"{diagnostic_criteria}\n"

    return result_string
