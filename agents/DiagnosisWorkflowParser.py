from typing import Union, List, Dict
import re

import pandas as pd
from langchain.agents import AgentOutputParser
from langchain.schema import AgentFinish
from thefuzz import process

from tools.Actions import Actions, is_valid_action
from tools.utils import (
    count_matches,
    count_radiology_modality_and_organ_matches,
    REGION_EXACT_DICT,
    REGION_SUBSTR_DICT,
    MODALITY_EXACT_DICT,
    MODALITY_SUBSTR_DICT,
    MODALITY_SPECIAL_CASES_DICT,
    UNIQUE_MODALITY_TO_ORGAN_MAPPING,
)
from agents.AgentAction import AgentAction
from utils.nlp import (
    # extract_keywords_spacy,
    extract_keywords_nltk,
    convert_labs_to_itemid,
    remove_stop_words,
)


class InvalidActionError(Exception):
    """Raised when an invalid action or action_input is provided in the LLM output."""

    invalid_tool_str = "Provide a diagnosis and treatment OR a valid tool. That"

    def __init__(self, llm_output, custom_parsings):
        self.invalid_agent_action = AgentAction(
            tool=self.invalid_tool_str,
            tool_input={"action_input": None},
            log=llm_output,
            custom_parsings=custom_parsings,
        )


class DiagnosisWorkflowParser(AgentOutputParser):
    lab_test_mapping_df: pd.DataFrame
    custom_parsings: int = 0
    action: str = ""
    action_input: Union[List[str], Dict] = None
    action_input_prepend: str = ""
    llm_output: str = ""

    class Config:
        arbitrary_types_allowed = True

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        self.llm_output = llm_output
        self.action_input = None
        self.action_input_prepend = ""
        self.action = ""
        self.custom_parsings = 0

        # Check if agent should finish
        if self.diagnosis_provided():
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": self.llm_output},
                log=self.llm_output,
            )

        try:
            # Parse action
            self.parse_action()

            # Interpret action
            self.interpret_action()

            # Check for action input if necessary
            if self.action in ["Imaging", "Laboratory Tests", "Diagnostic Criteria"]:
                self.parse_action_input()
        except InvalidActionError as e:
            return [e.invalid_agent_action]

        # Need to return as list because we have custom AgentAction class that is not automatically converted to list anymore in AgentExecutor
        return [
            AgentAction(
                tool=self.action,
                tool_input={"action_input": self.action_input},
                log=self.llm_output,
                custom_parsings=self.custom_parsings,
            )
        ]

    def diagnosis_provided(self) -> bool:
        """
        Check if 'diagnosis:' str is found in the LLM output to decide if FinalAction should be returned or normal parsing should be done.

        Returns:
            bool: True if 'diagnosis:' str is found in the LLM output, False otherwise
        """
        if "diagnosis:" in self.llm_output.lower():
            return True
        else:
            return False

    def parse_action(self) -> None:
        """
        Parse the desired action from the LLM output. Keep track of custom parsings done. If no explicit action is given, return invalid tool
        """
        # Parse the desired action)
        regex = r"Action:([\s\S]*?)(?=[\.\n].*Input.*:|$)"
        match = re.search(regex, self.llm_output, flags=re.IGNORECASE)

        if match:
            self.action = match.group(1).strip()
        else:
            # If no explicit action is given, return invalid tool. Completely free-text parsing is too instable
            raise InvalidActionError(self.llm_output, self.custom_parsings)

        # If action input is provided in action (e.g. after a dash)
        if "-" in self.action:
            split = self.action.split("-")
            self.action = split[0].strip()
            self.action_input_prepend = split[1].strip()
            self.custom_parsings += 1

        # Check for special cases
        if "labs" in self.action.lower():
            self.action = "Laboratory Tests"
            self.custom_parsings += 1
        elif "blood work" in self.action.lower():
            self.action = "Laboratory Tests"
            self.custom_parsings += 1

    def interpret_action(self) -> None:
        """
        Interpret the action provided by the LLM output. Check against valid actions to see if direct match in name. If not, try to pase as direct imaging or lab test desired i.e. action_input
        """
        # Check for action match
        action_match, match_score = process.extractOne(
            self.action, [e.value for e in Actions]
        )
        if match_score > 80:
            # Penalize if they didn't use the exact name
            if match_score != 100:
                self.custom_parsings += 1
            self.action = action_match
        else:
            # Check if specifics of action are directly given as action
            # i.e. 'Action: Abdominal Ultrasound' instead of 'Action: Imaging\nAction Input: Abdominal Ultrasound'
            self.custom_parsings += 1

            # Extract keywords, i.e. nouns and adjectives to understand if imaging or lab tests are directly given as action but just not in instructed format or if there is a large paragraph that happens to have some imaging/lab test keywords
            action_keywords = extract_keywords_nltk(self.action)
            # action_keywords = extract_keywords_spacy(action)

            # Check if imaging modality is directly given as action
            modality_counts = count_matches(
                self.action,
                exact_dict=MODALITY_EXACT_DICT,
                substr_dict=MODALITY_SUBSTR_DICT,
                special_cases_dict=MODALITY_SPECIAL_CASES_DICT,
            )
            organ_counts = count_matches(
                self.action,
                exact_dict=REGION_EXACT_DICT,
                substr_dict=REGION_SUBSTR_DICT,
            )

            # Valid imaging keywords should make up more than 25% of the action keywords or else it is likely a false positive
            if sum(modality_counts.values()) + sum(organ_counts.values()) > 0.25 * len(
                action_keywords
            ):
                self.action_input_prepend = self.action
                self.action = "Imaging"

            lab_test_names = self.lab_test_mapping_df["label"].tolist()

            # Remove single character tests (most importantly the test "I")
            lab_test_names = [name for name in lab_test_names if len(name) > 1]

            lab_test_matches = count_matches(
                self.action, exact_dict={"names": lab_test_names}
            )

            # Valid lab tests should make up more than 25% of the action or else it is likely a false positive
            if sum(lab_test_matches.values()) > 0.25 * len(action_keywords):
                self.action_input_prepend = self.action
                self.action = "Laboratory Tests"

        # If action is not part of valid actions, return invalid tool
        if not is_valid_action(self.action):
            raise InvalidActionError(self.llm_output, self.custom_parsings)

    def parse_action_input(self) -> None:
        """
        Parse the action input from the LLM output. If action input found, try to parse as imaging or lab tests. Diagnostic Criteria should be just the name of the disease. If not, return invalid tool
        """
        action_input_found = self.parse_action_input_from_llm_output()
        if action_input_found:
            if self.action == "Laboratory Tests":
                self.parse_lab_tests_action_input()
            elif self.action == "Imaging":
                try:
                    self.parse_imaging_action_input()
                except InvalidActionError:
                    raise
            elif self.action == "Diagnostic Criteria":
                self.parse_diagnostic_criteria_action_input()
            else:
                # Should never happen
                raise NotImplementedError

        # Imaging and laboratory tests require action inputs. If not provided, return invalid action
        if not self.action_input:
            raise InvalidActionError(self.llm_output, self.custom_parsings)

    def parse_action_input_from_llm_output(self) -> bool:
        """
        Parse the action input from the LLM output str

        Returns:
            bool: True if action input is found, False otherwise
        """
        regex = r"(Action )?Input:([\s\S]*)"
        match = re.search(regex, self.llm_output, flags=re.IGNORECASE)
        if match or self.action_input_prepend:
            if match:
                # Check if provided as 'Action Input' as instructed, or just 'Input'
                if not match.group(1):
                    self.custom_parsings += 1

                self.action_input = match.group(2)
            else:
                # Case where action + input is provided in action and we have it saved in action_input_prepend but no explicit 'Action Input' is given
                self.custom_parsings += 1
                self.action_input = ""

            # If input is none but info in action, use action as action_input i.e. Action: Abdominal Ultrasound\nAction Input: None
            if self.action_input == "None":
                self.action_input = ""

            self.action_input = (
                self.action_input_prepend + " " + self.action_input
            ).strip()
            return True
        return False

    def parse_imaging_action_input(self) -> None:
        """
        From extracted action input field, count number of matches of each modality and region and return most frequent
        """
        # Count matches of each modality and region and get most frequent plus counts
        (
            frequent_modality,
            frequent_modality_count,
            frequent_region,
            frequent_region_count,
        ) = count_radiology_modality_and_organ_matches(self.action_input)

        # if no region is found, check if a unique modality was given where the region is known
        if frequent_region_count == 0:
            if frequent_modality in UNIQUE_MODALITY_TO_ORGAN_MAPPING:
                frequent_region = UNIQUE_MODALITY_TO_ORGAN_MAPPING[frequent_modality]
                frequent_region_count = 1

        # If no modality or region is found, return invalid tool
        if frequent_region_count == 0 or frequent_modality_count == 0:
            raise InvalidActionError(self.llm_output, self.custom_parsings)
        else:
            self.action_input = {
                "modality": frequent_modality,
                "region": frequent_region,
            }

    def parse_lab_tests_action_input(self) -> None:
        """
        From the extracted action input field, seperate all desired lab tests into list and convert to canonical names
        """

        # Remove extra words from action input
        for word in ["order", "run", "level[s]?", "repeat", "check"]:
            change_check = self.action_input
            self.action_input = re.sub(
                rf"\b{word}\b", "", self.action_input, flags=re.IGNORECASE
            )
            if change_check != self.action_input:
                self.custom_parsings += 1

        # Replace and with comma to be found when splitting
        self.action_input = re.sub(r"\band\b", ",", self.action_input)

        # Replace new lines with comma to be found when splitting
        self.action_input = re.sub(r"\n", ",", self.action_input)

        # Remove stop words from action input
        self.action_input = remove_stop_words(self.action_input)

        # Convert to list by splitting on comma (except when between parantheses)
        self.action_input = re.split(r",\s*(?![^()]*\))", self.action_input)

        # Remove leading and trailing white spaces and remove entries that are only spaces (due to oxford comma replacement above)
        self.action_input = [
            a.strip() for a in self.action_input if a and not a.isspace()
        ]

        # Convert to canonical itemid found in dataset (identified through lab_test_mapping)
        self.action_input = convert_labs_to_itemid(
            self.action_input, self.lab_test_mapping_df
        )

        # Remove repeated entries
        self.action_input = [
            x
            for i, x in enumerate(self.action_input)
            if self.action_input.index(x) == i
        ]

    def parse_diagnostic_criteria_action_input(self) -> None:
        """
        From the extracted action input field, seperate all desired diagnostic criteria into list
        """

        # Replace and with comma to be found when splitting
        self.action_input = re.sub(r"\band\b", ",", self.action_input)

        # Replace new lines with comma to be found when splitting
        self.action_input = re.sub(r"\n", ",", self.action_input)

        # Remove stop words from action input
        self.action_input = remove_stop_words(self.action_input)

        # Convert to list by splitting on comma (except when between parantheses)
        self.action_input = re.split(r",\s*(?![^()]*\))", self.action_input)

        # Remove leading and trailing white spaces and remove entries that are only spaces (due to oxford comma replacement above)
        self.action_input = [
            a.strip() for a in self.action_input if a and not a.isspace()
        ]
