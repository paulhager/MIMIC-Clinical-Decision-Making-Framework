from evaluators.pathology_evaluator import PathologyEvaluator
from tools.utils import ADDITIONAL_LAB_TEST_MAPPING, INFLAMMATION_LAB_TESTS
from utils.nlp import (
    keyword_positive,
    procedure_checker,
    treatment_alternative_procedure_checker,
)
from icd.procedure_mappings import (
    CHOLECYSTECTOMY_PROCEDURES_ICD9,
    CHOLECYSTECTOMY_PROCEDURES_ICD10,
    CHOLECYSTECTOMY_PROCEDURES_KEYWORDS,
    ALTERNATE_CHOLECYSTECTOMY_KEYWORDS,
)


class CholecystitisEvaluator(PathologyEvaluator):
    """Evaluate the trajectory according to clinical diagnosis guidelines of cholecystitis."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pathology = "cholecystitis"
        self.alternative_pathology_names = [
            {
                "location": "gallbladder",
                "modifiers": [
                    "gangren",
                    "infect",
                    "inflam",
                    "abscess",
                    "necros",
                    "perf",
                ],
            },
            {
                "location": "cholangitis",
                "modifiers": [
                    "cholangitis",
                ],
            },
        ]
        self.gracious_alternative_pathology_names = [
            {"location": "acute gallbladder", "modifiers": ["disease", "attack"]},
            {"location": "acute biliary", "modifiers": ["colic"]},
        ]

        self.required_lab_tests = {
            "Inflammation": INFLAMMATION_LAB_TESTS,
            "Liver": [
                50861,  # "Alanine Aminotransferase (ALT)",
                50878,  # "Asparate Aminotransferase (AST)",
            ],
            "Gallbladder": [
                50883,  # "Bilirubin",
                50927,  # "Gamma Glutamyltransferase",
            ],
        }
        for req_lab_test_name in self.required_lab_tests:
            self.answers["Correct Laboratory Tests"][req_lab_test_name] = []

        self.neutral_lab_tests = []
        self.neutral_lab_tests.extend(
            ADDITIONAL_LAB_TEST_MAPPING["Complete Blood Count (CBC)"]
        )
        self.neutral_lab_tests.extend(
            ADDITIONAL_LAB_TEST_MAPPING["Renal Function Panel (RFP)"]
        )
        self.neutral_lab_tests.extend(
            [
                50863,  # "Alkaline Phosphatase"
            ]
        )
        self.neutral_lab_tests.extend(ADDITIONAL_LAB_TEST_MAPPING["Urinalysis"])
        self.neutral_lab_tests = [
            t
            for t in self.neutral_lab_tests
            if t not in self.required_lab_tests["Inflammation"]
            and t not in self.required_lab_tests["Liver"]
            and t not in self.required_lab_tests["Gallbladder"]
        ]

        self.answers["Treatment Requested"] = {
            "Cholecystectomy": False,
            "Antibiotics": False,
            "Support": False,
        }
        self.answers["Treatment Required"] = {
            "Cholecystectomy": False,
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
            # Preferred is regular US but MRI or endoscopic US is acceptable
            if modality == "Ultrasound" or modality == "HIDA":
                if self.scores["Imaging"] == 0:
                    self.scores["Imaging"] = 2
                return True
            if modality == "MRI" or modality == "EUS":
                if self.scores["Imaging"] == 0:
                    self.scores["Imaging"] = 1
                return True
        return False

    def score_treatment(self) -> None:
        ### CHOLECYSTECTOMY ###
        if (
            procedure_checker(CHOLECYSTECTOMY_PROCEDURES_ICD9, self.procedures_icd9)
            or procedure_checker(
                CHOLECYSTECTOMY_PROCEDURES_ICD10, self.procedures_icd10
            )
            or procedure_checker(
                CHOLECYSTECTOMY_PROCEDURES_KEYWORDS, self.procedures_discharge
            )
        ):
            self.answers["Treatment Required"]["Cholecystectomy"] = True

        if procedure_checker(
            CHOLECYSTECTOMY_PROCEDURES_KEYWORDS, [self.answers["Treatment"]]
        ) or treatment_alternative_procedure_checker(
            ALTERNATE_CHOLECYSTECTOMY_KEYWORDS, self.answers["Treatment"]
        ):
            self.answers["Treatment Requested"]["Cholecystectomy"] = True

        ### SUPPORT ###
        if (
            keyword_positive(self.answers["Treatment"], "fluid")
            or keyword_positive(self.answers["Treatment"], "analgesi")
            or keyword_positive(self.answers["Treatment"], "pain")
        ):
            self.answers["Treatment Requested"]["Support"] = True

        ### ANTIBIOTICS ###
        # TODO: Check antibiotics against medications table
        if keyword_positive(self.answers["Treatment"], "antibiotic"):
            self.answers["Treatment Requested"]["Antibiotics"] = True
