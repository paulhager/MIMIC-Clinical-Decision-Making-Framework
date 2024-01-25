from evaluators.pathology_evaluator import PathologyEvaluator
from tools.utils import ADDITIONAL_LAB_TEST_MAPPING, INFLAMMATION_LAB_TESTS
from utils.nlp import (
    keyword_positive,
    procedure_checker,
    treatment_alternative_procedure_checker,
)
from icd.procedure_mappings import (
    APPENDECTOMY_PROCEDURES_ICD9,
    APPENDECTOMY_PROCEDURES_ICD10,
    APPENDECTOMY_PROCEDURES_KEYWORDS,
    ALTERNATE_APPENDECTOMY_KEYWORDS,
)


class AppendicitisEvaluator(PathologyEvaluator):
    """Evaluate the trajectory according to clinical diagnosis guidelines of appendicitis."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pathology = "appendicitis"
        self.alternative_pathology_names = [
            {
                "location": "appendi",
                "modifiers": [
                    "gangren",
                    "infect",
                    "inflam",
                    "abscess",
                    "rupture",
                    "necros",
                    "perf",
                ],
            }
        ]
        self.gracious_alternative_pathology_names = []

        self.required_lab_tests = {
            "Inflammation": INFLAMMATION_LAB_TESTS,
        }
        for req_lab_test_name in self.required_lab_tests:
            self.answers["Correct Laboratory Tests"][req_lab_test_name] = []

        self.neutral_lab_tests = []
        self.neutral_lab_tests.extend(
            ADDITIONAL_LAB_TEST_MAPPING["Complete Blood Count (CBC)"]
        )
        self.neutral_lab_tests.extend(
            ADDITIONAL_LAB_TEST_MAPPING["Liver Function Panel (LFP)"]
        )
        self.neutral_lab_tests.extend(
            ADDITIONAL_LAB_TEST_MAPPING["Renal Function Panel (RFP)"]
        )
        self.neutral_lab_tests.extend(ADDITIONAL_LAB_TEST_MAPPING["Urinalysis"])
        self.neutral_lab_tests = [
            t
            for t in self.neutral_lab_tests
            if t not in self.required_lab_tests["Inflammation"]
        ]

        self.answers["Treatment Requested"] = {
            "Appendectomy": False,
            "Antibiotics": False,
            "Support": False,
        }
        self.answers["Treatment Required"] = {
            "Appendectomy": False,
            "Antibiotics": True,
            "Support": True,
        }

    def score_imaging(
        self,
        region: str,
        modality: str,
    ) -> None:
        # Region must be abdomen
        if region == "Abdomen":
            # TODO: Score according to what was done in case and not blindly following guidelines? i.e. if only CT was done by Dr, then give full points
            # Preferred imaging is US and should be done first
            if modality == "Ultrasound":
                if self.scores["Imaging"] == 0:
                    self.scores["Imaging"] = 2
                return True
            # CT is acceptable but should be done after US
            if modality == "CT":
                if self.scores["Imaging"] == 0:
                    self.scores["Imaging"] = 1
                return True
            # MRI is similar to CT and preferred for pregnant patients but should be done after US
            if modality == "MRI":
                if self.scores["Imaging"] == 0:
                    self.scores["Imaging"] = 1
                return True
        return False

    def score_treatment(self) -> None:
        ### APPENDECTOMY ###
        if (
            procedure_checker(APPENDECTOMY_PROCEDURES_ICD9, self.procedures_icd9)
            or procedure_checker(APPENDECTOMY_PROCEDURES_ICD10, self.procedures_icd10)
            or procedure_checker(
                APPENDECTOMY_PROCEDURES_KEYWORDS,
                self.procedures_discharge,
            )
        ):
            self.answers["Treatment Required"]["Appendectomy"] = True

        if procedure_checker(
            APPENDECTOMY_PROCEDURES_KEYWORDS, [self.answers["Treatment"]]
        ) or treatment_alternative_procedure_checker(
            ALTERNATE_APPENDECTOMY_KEYWORDS, self.answers["Treatment"]
        ):
            self.answers["Treatment Requested"]["Appendectomy"] = True

        ### ANTIBIOTICS ###
        # TODO: Check antibiotics against medications table
        if keyword_positive(self.answers["Treatment"], "antibiotic"):
            self.answers["Treatment Requested"]["Antibiotics"] = True

        ### SUPPORT ###
        if (
            keyword_positive(self.answers["Treatment"], "fluid")
            or keyword_positive(self.answers["Treatment"], "analgesi")
            or keyword_positive(self.answers["Treatment"], "pain")
        ):
            self.answers["Treatment Requested"]["Support"] = True
