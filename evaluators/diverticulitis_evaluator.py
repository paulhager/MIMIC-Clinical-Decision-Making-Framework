# import spacy

from evaluators.pathology_evaluator import PathologyEvaluator
from utils.nlp import (
    procedure_checker,
    keyword_positive,
    treatment_alternative_procedure_checker,
)
from tools.utils import ADDITIONAL_LAB_TEST_MAPPING, INFLAMMATION_LAB_TESTS
from icd.procedure_mappings import (
    COLECTOMY_PROCEDURES_ICD10,
    COLECTOMY_PROCEDURES_ICD9,
    COLECTOMY_PROCEDURES_KEYWORDS,
    ALTERNATE_COLECTOMY_KEYWORDS,
    DRAINAGE_PROCEDURES_ICD9,
    DRAINAGE_PROCEDURES_ALL_ICD10,
    DRAINAGE_PROCEDURES_KEYWORDS,
    DRAINAGE_LOCATIONS_DIVERTICULITIS,
    ALTERNATE_DRAINAGE_KEYWORDS_DIVERTICULITIS,
)


class DiverticulitisEvaluator(PathologyEvaluator):
    """Evaluate the trajectory according to clinical diagnosis guidelines of diverticulitis."""

    # nlp = spacy.load("en_core_web_sm")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pathology = "diverticulitis"
        self.alternative_pathology_names = [
            {
                "location": "diverticul",
                "modifiers": ["inflam", "infect", "abscess", "perf", "rupture"],
            },
        ]
        self.gracious_alternative_pathology_names = [
            {
                "location": "acute colonic",
                "modifiers": ["perfor"],
            },
            {
                "location": "sigmoid",
                "modifiers": ["perfor"],
            },
            {
                "location": "sigmoid",
                "modifiers": ["colitis"],
            },
        ]

        self.required_lab_tests = {"Inflammation": INFLAMMATION_LAB_TESTS}
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
            "Colonoscopy": False,
            "Antibiotics": False,
            "Support": False,
            "Drainage": False,
            "Colectomy": False,
        }
        self.answers["Treatment Required"] = {
            "Colonoscopy": True,
            "Antibiotics": True,
            "Support": True,
            "Drainage": False,
            "Colectomy": False,
        }

    def score_imaging(
        self,
        region: str,
        modality: str,
    ) -> None:
        # Region must be abdomen
        if region == "Abdomen":
            # CT is preferred but US accepted
            if modality == "CT":
                if self.scores["Imaging"] == 0:
                    self.scores["Imaging"] = 2
                return True
            if modality == "Ultrasound" or modality == "MRI":
                if self.scores["Imaging"] == 0:
                    self.scores["Imaging"] = 1
                return True
        return False

    def check_colonoscopy_time_order(self) -> bool:
        doc = self.nlp(self.answers["Treatment"].lower())

        colonoscopy = False
        after_treatment = False
        for token in doc:
            if token.text == "colonoscopy":
                colonoscopy = True

            if (
                token.text
                in [
                    "treatment",
                    "curation",
                    "treating",
                    "curing",
                    "episode",
                    "course",
                    "case",
                    "resolution",
                    "therapy",
                    "intervention",
                    "management",
                    "remedy",
                    "amelioration",
                    "relief",
                    "eradication",
                    "rehabilitation",
                    "mitigation",
                    "resolution",
                    "control",
                    "recovery",
                    "containment",
                    "elimination",
                    "interruption",
                    "suppression",
                    "stabilization",
                    "outcome",
                    "conclusion",
                    "settlement",
                ]
                and token.head.text.lower() == "after"
            ):
                after_treatment = True

        if colonoscopy and after_treatment:
            return True

    def score_treatment(self) -> None:
        ### COLONOSCOPY ###
        # TODO: Must be done AFTER curation to check for cancer. During acute pathology can lead to perforation and is AGAINST the guidelines. Temporal order difficult to check automatically. Currently confirmed manually
        # if self.check_colonoscopy_time_order():
        if keyword_positive(self.answers["Treatment"], "colonoscopy"):
            self.answers["Treatment Requested"]["Colonoscopy"] = True

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

        ### DRAINAGE ###
        if (
            procedure_checker(DRAINAGE_PROCEDURES_ICD9, self.procedures_icd9)
            or procedure_checker(DRAINAGE_PROCEDURES_ALL_ICD10, self.procedures_icd10)
            or (
                procedure_checker(
                    DRAINAGE_PROCEDURES_KEYWORDS, self.procedures_discharge
                )
                and procedure_checker(
                    DRAINAGE_LOCATIONS_DIVERTICULITIS, self.procedures_discharge
                )
            )
        ):
            self.answers["Treatment Required"]["Drainage"] = True

        if treatment_alternative_procedure_checker(
            ALTERNATE_DRAINAGE_KEYWORDS_DIVERTICULITIS, self.answers["Treatment"]
        ):
            self.answers["Treatment Requested"]["Drainage"] = True

        ### COLECTOMY ###
        if (
            procedure_checker(COLECTOMY_PROCEDURES_ICD9, self.procedures_icd9)
            or procedure_checker(COLECTOMY_PROCEDURES_ICD10, self.procedures_icd10)
            or procedure_checker(
                COLECTOMY_PROCEDURES_KEYWORDS, self.procedures_discharge
            )
        ):
            self.answers["Treatment Required"]["Colectomy"] = True

        if procedure_checker(
            COLECTOMY_PROCEDURES_KEYWORDS, [self.answers["Treatment"]]
        ) or treatment_alternative_procedure_checker(
            ALTERNATE_COLECTOMY_KEYWORDS, self.answers["Treatment"]
        ):
            self.answers["Treatment Requested"]["Colectomy"] = True
