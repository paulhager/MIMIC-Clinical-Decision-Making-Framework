import re
from typing import Any, Optional, Sequence, Tuple, List, Dict
from abc import abstractmethod
from torch import Tensor

from thefuzz import fuzz

from langchain.evaluation import AgentTrajectoryEvaluator
from agents.AgentAction import AgentAction
from utils.nlp import keyword_positive, remove_punctuation
from agents.DiagnosisWorkflowParser import InvalidActionError
from models.utils import calculate_log_prob_confidence


class PathologyEvaluator(AgentTrajectoryEvaluator):
    """Evaluate the trajectory according to clinical diagnosis guidelines of <PATHOLOGY>."""

    pathology: str
    alternative_pathology_names: List[Dict]
    gracious_alternative_pathology_names: List[Dict]
    required_lab_tests: Dict[str, List[str]]
    neutral_lab_tests: List[str]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.answers = {
            "Diagnosis": "",
            "Diagnostic Confidence": None,
            "Treatment": "",
            "Unnecessary Laboratory Tests": [],
            "Correct Laboratory Tests": {},
            "Unnecessary Imaging": [],
            "Correct Imaging": [],
        }
        self.scores = {
            "Late Physical Examination": 0,
            "Physical Examination": 0,
            "Laboratory Tests": 0,
            "Imaging": 0,
            "Diagnosis": 0,
            "Gracious Diagnosis": 0,
            "Action Parsing": 0,
            "Treatment Parsing": 0,
            "Diagnosis Parsing": 0,
            "Invalid Tools": 0,
            "Rounds": 0,
        }

    def _evaluate_agent_trajectory(
        self,
        *,
        prediction: str,
        input: str,
        agent_trajectory: Sequence[Tuple[AgentAction, str]],
        reference: Optional[
            Tuple[str, List[str], List[str], List[str], List[str], List[str]]
        ] = None,
        diagnosis_probabilities: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> dict:
        self.discharge_diagnosis = reference[0]
        self.icd_diagnoses = reference[1]
        self.procedures_icd9 = reference[2]
        self.procedures_icd10 = reference[3]
        self.procedures_discharge = reference[4]

        for indx, actions in enumerate(agent_trajectory):
            action, observation = actions

            # Keep running tally of custom parsings done
            if action.custom_parsings > 0:
                self.scores["Action Parsing"] = 1

            # Keep running tally of invalid tool requests
            if action.tool == InvalidActionError.invalid_tool_str:
                self.scores["Invalid Tools"] += 1

            # First action should be physical examination
            if action.tool == "Physical Examination":
                self.score_physical_examination(action, indx)

            if action.tool == "Laboratory Tests":
                self.score_laboratory_tests(action)

            if action.tool == "Imaging":
                self.score_imaging_action(action)

        # Record number of rounds needed to reach diagnosis
        self.scores["Rounds"] = len(agent_trajectory)

        # Parse and score diagnosis (and confidence if available)
        self.parse_diagnosis(prediction)
        self.score_diagnosis()
        if diagnosis_probabilities is not None:
            self.answers["Diagnostic Confidence"] = diagnosis_probabilities.squeeze()

        # Parse and score treatment
        if len(agent_trajectory) > 0:
            self.parse_treatment(prediction)
            self.score_treatment()

        return {
            "scores": self.scores,
            "answers": self.answers,
        }

    def parse_treatment(self, prediction: str) -> None:
        """Takes prediction string and parses it for treatment. Checks how well the LLM held itself to the desired format.

        Args:
            prediction (str): The prediction string.
        """
        regex = r"Treatment(.*)?:(.*)"
        match = re.search(regex, prediction, flags=re.DOTALL | re.IGNORECASE)
        if match:
            # Instructions were to only write Treatment. If not followed, penalize
            if match.group(1):
                self.scores["Treatment Parsing"] = 1

            treatment = match.group(2).strip()
            self.answers["Treatment"] = treatment

    def parse_diagnosis(self, prediction: str) -> str:
        """Takes prediction string and parses it for diagnosis. Checks how well the LLM held itself to the desired format.

        Args:
            prediction (str): The prediction string.
        """
        custom_parsing = False
        regex = r"(Final )?Diagnosis:([\s\S]*?)(?=[\.\n].*Treatment.*:|$)"
        match = re.search(regex, prediction, flags=re.IGNORECASE)
        if match:
            # Instructions were to write Final Diagnosis. If not followed, penalize
            if not match.group(1):
                custom_parsing = True

            diagnosis = match.group(2).strip()

            # Llama 2 Chat has an intro sentence we want to remove
            modify_check = diagnosis
            diagnosis = re.sub(r"^Based on.*:\n\n", "", diagnosis)
            if modify_check != diagnosis:
                custom_parsing = True

            # Delete extra sections so they are not parsed as diagnosis accidentally
            modify_check = diagnosis
            for section in [
                "rationale",
                "note",
                "recommendation",
                "explanation",
                "finding",
                "other.*diagnos.*include",
                "other.*diagnos.*considered(?: were)?",
                "management",
                "action",
                "plan",
                "reasoning",
                "assessment",
                "justification",
                "tests",
                "additional diagnoses",
                "notification",
                "impression",
                "background",
                "additional findings include",
            ]:
                diagnosis = re.sub(
                    rf"{section}[s]?:.*", "", diagnosis, flags=re.IGNORECASE | re.DOTALL
                )
            if modify_check != diagnosis:
                custom_parsing = True

            # Check for lists with numbers. Extract the first entry
            modify_check = diagnosis
            match = re.search(r"^1\.(.*)", diagnosis, flags=re.MULTILINE)
            if match:
                diagnosis = match.group(1).strip()
            if modify_check != diagnosis:
                custom_parsing = True
                # Remove unwanted explanations
                diagnosis = re.sub(r"[-:].*", "", diagnosis)

            # Check for lists using stars. Extract the first entry
            modify_check = diagnosis
            match = re.search(r"^\*(.*)", diagnosis, flags=re.MULTILINE)
            if match:
                diagnosis = match.group(1).strip()
            if modify_check != diagnosis:
                custom_parsing = True
                # Remove unwanted explanations
                diagnosis = re.sub(r"[-:] .*", "", diagnosis)

            # Llama 2 Chat often does a double newline with an explanation after the diagnosis
            modify_check = diagnosis
            diagnosis = re.sub(r"\n\n.*", "", diagnosis)
            if modify_check != diagnosis:
                custom_parsing = True

            # Check for diagnosis contained in the sentence
            modify_check = diagnosis
            match = re.search(
                r".*?diagnosis[^.\n]*?\bis\b(.*?)[.\n]", diagnosis, flags=re.DOTALL
            )
            if match:
                diagnosis = match.group(1).strip()
            if modify_check != diagnosis:
                custom_parsing = True

            # Check for diagnosis contained in the sentence
            modify_check = diagnosis
            diagnosis = re.sub(
                r".*?patient has", "", diagnosis, count=1, flags=re.DOTALL
            )
            if modify_check != diagnosis:
                custom_parsing = True

            # Check for multiple diagnoses.
            diagnoses = re.split(r"[,.\n]|(?:\s*\b(?:and|or|vs[.]?)\b\s*)", diagnosis)
            diagnoses = [d for d in diagnoses if d != ""]

            # Instructions were to just write a single diagnosis. If not followed, penalize
            if len(diagnoses) > 1:
                custom_parsing = True

            # We do not allow for more than two diagnoses. Indicated great uncertainty and task was to provide a single diagnosis
            # if len(diagnoses) > 2 or len(diagnoses) == 0:
            if len(diagnoses) == 0:
                diagnosis = ""
            # If multiple diagnoses provided, take first as the more likely diagnosis and use for evaluation purposes
            else:
                diagnosis = diagnoses[0]

            if custom_parsing:
                self.scores["Diagnosis Parsing"] = 1

            self.answers["Diagnosis"] = diagnosis.strip()

    def score_diagnosis(self) -> None:
        """Checks predicted pathology against that of the patient. Ensures diagnosis does not contain negation"""
        answer = remove_punctuation(self.answers["Diagnosis"].lower())
        for word in answer.split():
            if fuzz.ratio(word, self.pathology) > 90 and keyword_positive(
                self.answers["Diagnosis"], word
            ):
                self.scores["Diagnosis"] = 1
                self.scores["Gracious Diagnosis"] = 1
                break
        for alternative_patho in self.alternative_pathology_names:
            patho_loc = alternative_patho["location"]
            for patho_mod in alternative_patho["modifiers"]:
                if (
                    patho_loc in answer
                    and patho_mod in answer
                    and keyword_positive(self.answers["Diagnosis"], patho_loc)
                    and keyword_positive(self.answers["Diagnosis"], patho_mod)
                ):
                    self.scores["Diagnosis"] = 1
                    self.scores["Gracious Diagnosis"] = 1
                    break
        # Can be more or less gracious with what is accepted as a correct diagnosis
        for alternative_patho in self.gracious_alternative_pathology_names:
            patho_loc = alternative_patho["location"]
            for patho_mod in alternative_patho["modifiers"]:
                if (
                    patho_loc in answer
                    and patho_mod in answer
                    and keyword_positive(self.answers["Diagnosis"], patho_loc)
                    and keyword_positive(self.answers["Diagnosis"], patho_mod)
                ):
                    self.scores["Gracious Diagnosis"] = 1
                    break

    def score_physical_examination(self, action: AgentAction, indx: int) -> None:
        """A physical examination must be done. It should be done first. Full points only awarded if done first

        Args:
          action (AgentAction): The physical examination action.
          indx (int): The index of the action in the trajectory.
        """
        if indx == 0:
            self.scores["Physical Examination"] = 1
            self.scores["Late Physical Examination"] = 1
        else:
            self.scores["Late Physical Examination"] = 1

    def score_laboratory_tests(self, action: AgentAction) -> None:
        """Score all laboratory tests according to diagnostic guidelines. Each category of test should be performed at least once. Multiple tests in the same category are ignored. Unnecessary tests (that are not neutral i.e. used for eliminating differential diagnoses) are recorded.

        Args:
          action (AgentAction): The laboratory test action.
        """
        for test in action.tool_input["action_input"]:
            for test_category, valid_test_names in self.required_lab_tests.items():
                if test in valid_test_names:
                    # Only provide points for first test occurance in each category
                    if (
                        len(self.answers["Correct Laboratory Tests"][test_category])
                        == 0
                    ):
                        self.scores["Laboratory Tests"] += 1
                    self.answers["Correct Laboratory Tests"][test_category].append(test)
                    break
            else:
                if test not in self.neutral_lab_tests:
                    self.answers["Unnecessary Laboratory Tests"].append(test)

    def score_imaging_action(
        self,
        action: AgentAction,
    ) -> None:
        region = action.tool_input["action_input"]["region"]
        modality = action.tool_input["action_input"]["modality"]
        imaging_dict = {"region": region, "modality": modality}

        # Run patho specific scoring and check if valid modality + region combination
        valid_imaging_combination = self.score_imaging(region, modality)
        if (
            not valid_imaging_combination
            or imaging_dict in self.answers["Correct Imaging"]
        ):
            self.answers["Unnecessary Imaging"].append(imaging_dict)
        else:
            self.answers["Correct Imaging"].append(imaging_dict)

    @abstractmethod
    def score_imaging(self, region: str, modality: str) -> None:
        """Score all imaging tests according to diagnostic guidelines. Unnecessary tests are recorded. Also check if valid modality + region combination.

        Args:
          region (str): The region of the body to be scanned.
          modality (str): The imaging modality.

        Returns:
          bool: True if valid modality + region combination, False otherwise.
        """

    @abstractmethod
    def score_treatment(self) -> None:
        """Score a treatment according to diagnostic guidelines."""
