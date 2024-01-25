import unittest
from evaluators.diverticulitis_evaluator import DiverticulitisEvaluator
from agents.AgentAction import AgentAction


class TestDiverticulitisEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = DiverticulitisEvaluator()
        self.maxDiff = None
        self.evaluator.discharge_diagnosis = "Acute Diverticulitis"
        self.evaluator.icd_diagnoses = [
            "Diverticulitis of large intestine without perforation and abscess without bleeding"
        ]
        self.evaluator.procedures_icd9 = []
        self.evaluator.procedures_icd10 = []
        self.evaluator.procedures_discharge = []

    ##############
    # LABORATORY #
    ##############

    def test_score_laboratory_tests_inflammation(self):
        test_ordered = [51301]  # White Blood Cells
        action = AgentAction(
            tool="Laboratory Tests",
            tool_input={"action_input": test_ordered},
            log="",
            custom_parsings=0,
        )

        self.evaluator.score_laboratory_tests(action)

        self.assertEqual(self.evaluator.scores["Laboratory Tests"], 1)
        self.assertEqual(
            self.evaluator.answers["Correct Laboratory Tests"],
            {"Inflammation": [51301]},
        )
        self.assertEqual(self.evaluator.answers["Unnecessary Laboratory Tests"], [])

    def test_score_laboratory_tests_neutral_cbc(self):
        test_ordered = [51279]  # Red Blood Cells
        action = AgentAction(
            tool="Laboratory Tests",
            tool_input={"action_input": test_ordered},
            log="",
            custom_parsings=0,
        )

        self.evaluator.score_laboratory_tests(action)

        self.assertEqual(self.evaluator.scores["Laboratory Tests"], 0)
        self.assertEqual(
            self.evaluator.answers["Correct Laboratory Tests"], {"Inflammation": []}
        )
        self.assertEqual(self.evaluator.answers["Unnecessary Laboratory Tests"], [])

    def test_score_laboratory_tests_neutral_lfp(self):
        test_ordered = [50861]  # ALT
        action = AgentAction(
            tool="Laboratory Tests",
            tool_input={"action_input": test_ordered},
            log="",
            custom_parsings=0,
        )

        self.evaluator.score_laboratory_tests(action)

        self.assertEqual(self.evaluator.scores["Laboratory Tests"], 0)
        self.assertEqual(
            self.evaluator.answers["Correct Laboratory Tests"], {"Inflammation": []}
        )
        self.assertEqual(self.evaluator.answers["Unnecessary Laboratory Tests"], [])

    def test_score_laboratory_tests_neutral_rfp(self):
        test_ordered = [50862]  # Albumin
        action = AgentAction(
            tool="Laboratory Tests",
            tool_input={"action_input": test_ordered},
            log="",
            custom_parsings=0,
        )

        self.evaluator.score_laboratory_tests(action)

        self.assertEqual(self.evaluator.scores["Laboratory Tests"], 0)
        self.assertEqual(
            self.evaluator.answers["Correct Laboratory Tests"], {"Inflammation": []}
        )
        self.assertEqual(self.evaluator.answers["Unnecessary Laboratory Tests"], [])

    def test_score_laboratory_tests_neutral_urine(self):
        test_ordered = [51508]  # Urine Color
        action = AgentAction(
            tool="Laboratory Tests",
            tool_input={"action_input": test_ordered},
            log="",
            custom_parsings=0,
        )

        self.evaluator.score_laboratory_tests(action)

        self.assertEqual(self.evaluator.scores["Laboratory Tests"], 0)
        self.assertEqual(
            self.evaluator.answers["Correct Laboratory Tests"], {"Inflammation": []}
        )
        self.assertEqual(self.evaluator.answers["Unnecessary Laboratory Tests"], [])

    ###########
    # IMAGING #
    ###########

    def test_score_imaging_diverticulitis_US(self):
        region = "Abdomen"
        modality = "Ultrasound"

        self.evaluator.score_imaging(region, modality)

        self.assertEqual(self.evaluator.scores["Imaging"], 1)

    def test_score_imaging_diverticulitis_CT(self):
        region = "Abdomen"
        modality = "CT"

        self.evaluator.score_imaging(region, modality)

        self.assertEqual(self.evaluator.scores["Imaging"], 2)

    def test_score_imaging_diverticulitis_MRI(self):
        region = "Abdomen"
        modality = "MRI"

        self.evaluator.score_imaging(region, modality)

        self.assertEqual(self.evaluator.scores["Imaging"], 1)

    def test_score_imaging_diverticulitis_wrong_region(self):
        region = "Head"
        modality = "MRI"

        self.evaluator.score_imaging(region, modality)

        self.assertEqual(self.evaluator.scores["Imaging"], 0)

    def test_score_imaging_diverticulitis_CT_then_US(self):
        region = "Abdomen"
        modality = "CT"

        self.evaluator.score_imaging(region, modality)

        region = "Abdomen"
        modality = "Ultrasound"

        self.evaluator.score_imaging(region, modality)

        self.assertEqual(self.evaluator.scores["Imaging"], 2)

    def test_score_imaging_diverticulitis_US_then_CT(self):
        region = "Abdomen"
        modality = "Ultrasound"

        self.evaluator.score_imaging(region, modality)

        region = "Abdomen"
        modality = "CT"

        self.evaluator.score_imaging(region, modality)

        self.assertEqual(self.evaluator.scores["Imaging"], 1)

    def test_score_imaging_diverticulitis_repeat(self):
        region = "Abdomen"
        modality = "Ultrasound"

        self.evaluator.score_imaging(region, modality)

        region = "Abdomen"
        modality = "Ultrasound"

        self.evaluator.score_imaging(region, modality)

        self.assertEqual(self.evaluator.scores["Imaging"], 1)

    def test_score_imaging_action_diverticulitis_US(self):
        imaging_dict = {"region": "Abdomen", "modality": "Ultrasound"}
        action = AgentAction(
            tool="Imaging",
            tool_input={"action_input": imaging_dict},
            log="",
            custom_parsings=0,
        )

        self.evaluator.score_imaging_action(action)

        self.assertEqual(self.evaluator.answers["Correct Imaging"], [imaging_dict])
        self.assertEqual(self.evaluator.answers["Unnecessary Imaging"], [])

    def test_score_imaging_action_diverticulitis_CT(self):
        imaging_dict = {"region": "Abdomen", "modality": "CT"}
        action = AgentAction(
            tool="Imaging",
            tool_input={"action_input": imaging_dict},
            log="",
            custom_parsings=0,
        )

        self.evaluator.score_imaging_action(action)

        self.assertEqual(self.evaluator.answers["Correct Imaging"], [imaging_dict])
        self.assertEqual(self.evaluator.answers["Unnecessary Imaging"], [])

    def test_score_imaging_action_diverticulitis_MRI(self):
        imaging_dict = {"region": "Abdomen", "modality": "MRI"}
        action = AgentAction(
            tool="Imaging",
            tool_input={"action_input": imaging_dict},
            log="",
            custom_parsings=0,
        )

        self.evaluator.score_imaging_action(action)

        self.assertEqual(self.evaluator.answers["Correct Imaging"], [imaging_dict])
        self.assertEqual(self.evaluator.answers["Unnecessary Imaging"], [])

    def test_score_imaging_action_diverticulitis_wrong_region(self):
        imaging_dict = {"region": "Head", "modality": "MRI"}
        action = AgentAction(
            tool="Imaging",
            tool_input={"action_input": imaging_dict},
            log="",
            custom_parsings=0,
        )

        self.evaluator.score_imaging_action(action)

        self.assertEqual(self.evaluator.answers["Correct Imaging"], [])
        self.assertEqual(self.evaluator.answers["Unnecessary Imaging"], [imaging_dict])

    def test_score_imaging_action_diverticulitis_CT_then_US(self):
        imaging_dict_ct = {"region": "Abdomen", "modality": "CT"}
        action_ct = AgentAction(
            tool="Imaging",
            tool_input={"action_input": imaging_dict_ct},
            log="",
            custom_parsings=0,
        )

        self.evaluator.score_imaging_action(action_ct)

        imaging_dict_us = {"region": "Abdomen", "modality": "Ultrasound"}
        action_us = AgentAction(
            tool="Imaging",
            tool_input={"action_input": imaging_dict_us},
            log="",
            custom_parsings=0,
        )

        self.evaluator.score_imaging_action(action_us)

        self.assertEqual(
            self.evaluator.answers["Correct Imaging"],
            [imaging_dict_ct, imaging_dict_us],
        )

    def test_score_imaging_action_diverticulitis_US_then_CT(self):
        imaging_dict_us = {"region": "Abdomen", "modality": "Ultrasound"}
        action_us = AgentAction(
            tool="Imaging",
            tool_input={"action_input": imaging_dict_us},
            log="",
            custom_parsings=0,
        )

        self.evaluator.score_imaging_action(action_us)

        imaging_dict_ct = {"region": "Abdomen", "modality": "CT"}
        action_ct = AgentAction(
            tool="Imaging",
            tool_input={"action_input": imaging_dict_ct},
            log="",
            custom_parsings=0,
        )

        self.evaluator.score_imaging_action(action_ct)

        self.assertEqual(
            self.evaluator.answers["Correct Imaging"],
            [imaging_dict_us, imaging_dict_ct],
        )

    def test_score_imaging_action_diverticulitis_repeat(self):
        imaging_dict = {"region": "Abdomen", "modality": "Ultrasound"}
        action = AgentAction(
            tool="Imaging",
            tool_input={"action_input": imaging_dict},
            log="",
            custom_parsings=0,
        )

        self.evaluator.score_imaging_action(action)

        self.evaluator.score_imaging_action(action)

        self.assertEqual(self.evaluator.answers["Correct Imaging"], [imaging_dict])
        self.assertEqual(self.evaluator.answers["Unnecessary Imaging"], [imaging_dict])

    #############
    # DIAGNOSIS #
    #############

    def test_evaluator_diagnosis(self):
        result = {
            "input": "",
            "output": "Final Diagnosis: Acute Diverticulitis",
            "intermediate_steps": [],
        }

        eval = self.evaluator._evaluate_agent_trajectory(
            prediction=result["output"],
            input=result["input"],
            reference=(
                "Acute Diverticulitis",
                self.evaluator.icd_diagnoses,
                [],
                [],
                [],
            ),
            agent_trajectory=result["intermediate_steps"],
        )

        self.assertEqual(eval["scores"]["Diagnosis"], 1)
        self.assertEqual(
            eval["answers"]["Diagnosis"], self.evaluator.discharge_diagnosis
        )

    def test_score_diagnosis_acute_diverticulitis(self):
        self.evaluator.answers["Diagnosis"] = "Acute diverticulitis"
        self.evaluator.score_diagnosis()

        self.assertEqual(self.evaluator.scores["Diagnosis"], 1)

    def test_score_diagnosis_diverticulitis(self):
        self.evaluator.answers["Diagnosis"] = "diverticulitis"
        self.evaluator.score_diagnosis()

        self.assertEqual(self.evaluator.scores["Diagnosis"], 1)

    def test_score_diagnosis_inflammation_diverticula(self):
        self.evaluator.answers["Diagnosis"] = "Inflammed diverticula"
        self.evaluator.score_diagnosis()

        self.assertEqual(self.evaluator.scores["Diagnosis"], 1)

    def test_score_diagnosis_inflammation_diverticula_od(self):
        self.evaluator.answers["Diagnosis"] = "Inflammation of the diverticula"
        self.evaluator.score_diagnosis()

        self.assertEqual(self.evaluator.scores["Diagnosis"], 1)

    def test_score_diagnosis_colonic_diverticulitis(self):
        self.evaluator.answers["Diagnosis"] = "Colonic diverticulitis"
        self.evaluator.score_diagnosis()

        self.assertEqual(self.evaluator.scores["Diagnosis"], 1)

    def test_score_diagnosis_intestinal_diverticulitis(self):
        self.evaluator.answers["Diagnosis"] = "Intestinal diverticulitis"
        self.evaluator.score_diagnosis()

        self.assertEqual(self.evaluator.scores["Diagnosis"], 1)

    def test_score_diagnosis_large_intestine_diverticulitis(self):
        self.evaluator.answers["Diagnosis"] = "Diverticulitis of the large intestine"
        self.evaluator.score_diagnosis()

        self.assertEqual(self.evaluator.scores["Diagnosis"], 1)

    def test_score_diagnosis_large_intestine_diverticula(self):
        self.evaluator.answers["Diagnosis"] = "Diverticula"
        self.evaluator.score_diagnosis()

        self.assertEqual(self.evaluator.scores["Diagnosis"], 0)

    def test_score_diagnosis_large_intestine_unspecific(self):
        self.evaluator.answers["Diagnosis"] = "Diverticulosis"
        self.evaluator.score_diagnosis()

        self.assertEqual(self.evaluator.scores["Diagnosis"], 0)

    def test_score_diagnosis_large_intestine_unspecific_2(self):
        self.evaluator.answers["Diagnosis"] = "Diverticulum"
        self.evaluator.score_diagnosis()

        self.assertEqual(self.evaluator.scores["Diagnosis"], 0)

    ##############
    # TREATMENT  #
    ##############

    def test_evaluator_treatment_diverticulitis(self):
        action = AgentAction(
            tool="",
            tool_input={},
            log="",
            custom_parsings=0,
        )
        result = {
            "input": "",
            "output": "Diagnosis: Diverticulitis\nTreatment: Antibiotics, Fluids, Analgesia, Monitoring",
            "intermediate_steps": [(action, "")],
        }

        eval = self.evaluator._evaluate_agent_trajectory(
            prediction=result["output"],
            input=result["input"],
            reference=(
                self.evaluator.discharge_diagnosis,
                self.evaluator.icd_diagnoses,
                self.evaluator.procedures_icd9,
                self.evaluator.procedures_icd10,
                self.evaluator.procedures_discharge,
            ),
            agent_trajectory=result["intermediate_steps"],
        )

        self.assertEqual(
            self.evaluator.answers["Treatment Requested"],
            {
                "Colonoscopy": False,
                "Antibiotics": True,
                "Support": True,
                "Drainage": False,
                "Colectomy": False,
            },
        )

        self.assertEqual(
            self.evaluator.answers["Treatment Required"],
            {
                "Colonoscopy": True,
                "Antibiotics": True,
                "Support": True,
                "Drainage": False,
                "Colectomy": False,
            },
        )

        self.assertEqual(
            eval["answers"]["Treatment"],
            "Antibiotics, Fluids, Analgesia, Monitoring",
        )

    def test_max_scores_colonoscopy(self):
        self.evaluator.answers["Treatment"] = "Colonoscopy"
        self.evaluator.score_treatment()

        self.assertEqual(
            self.evaluator.answers["Treatment Requested"],
            {
                "Colonoscopy": True,
                "Antibiotics": False,
                "Support": False,
                "Drainage": False,
                "Colectomy": False,
            },
        )

        self.assertEqual(
            self.evaluator.answers["Treatment Required"],
            {
                "Colonoscopy": True,
                "Antibiotics": True,
                "Support": True,
                "Drainage": False,
                "Colectomy": False,
            },
        )

    def test_score_treatment_antibiotics(self):
        self.evaluator.answers["Treatment"] = "Antibiotics"
        self.evaluator.score_treatment()

        self.assertEqual(
            self.evaluator.answers["Treatment Requested"],
            {
                "Colonoscopy": False,
                "Antibiotics": True,
                "Support": False,
                "Drainage": False,
                "Colectomy": False,
            },
        )

        self.assertEqual(
            self.evaluator.answers["Treatment Required"],
            {
                "Colonoscopy": True,
                "Antibiotics": True,
                "Support": True,
                "Drainage": False,
                "Colectomy": False,
            },
        )

    def test_score_treatment_support(self):
        self.evaluator.answers["Treatment"] = "Fluids, Analgesia, Monitoring"
        self.evaluator.score_treatment()

        self.assertEqual(
            self.evaluator.answers["Treatment Requested"],
            {
                "Colonoscopy": False,
                "Antibiotics": False,
                "Support": True,
                "Drainage": False,
                "Colectomy": False,
            },
        )

        self.assertEqual(
            self.evaluator.answers["Treatment Required"],
            {
                "Colonoscopy": True,
                "Antibiotics": True,
                "Support": True,
                "Drainage": False,
                "Colectomy": False,
            },
        )

    def test_score_treatment_drainage_icd9(self):
        self.evaluator.procedures_icd9 = [5491]
        self.evaluator.answers["Treatment"] = "Diverticular drainage"
        self.evaluator.score_treatment()

        self.assertEqual(
            self.evaluator.answers["Treatment Requested"],
            {
                "Colonoscopy": False,
                "Antibiotics": False,
                "Support": False,
                "Drainage": True,
                "Colectomy": False,
            },
        )

        self.assertEqual(
            self.evaluator.answers["Treatment Required"],
            {
                "Colonoscopy": True,
                "Antibiotics": True,
                "Support": True,
                "Drainage": True,
                "Colectomy": False,
            },
        )

    def test_score_treatment_drainage_icd10(self):
        self.evaluator.procedures_icd10 = ["0W9G30Z"]
        self.evaluator.answers["Treatment"] = "Pericolonic drainage"
        self.evaluator.score_treatment()

        self.assertEqual(
            self.evaluator.answers["Treatment Requested"],
            {
                "Colonoscopy": False,
                "Antibiotics": False,
                "Support": False,
                "Drainage": True,
                "Colectomy": False,
            },
        )

        self.assertEqual(
            self.evaluator.answers["Treatment Required"],
            {
                "Colonoscopy": True,
                "Antibiotics": True,
                "Support": True,
                "Drainage": True,
                "Colectomy": False,
            },
        )

    def test_score_treatment_drainage_discharge(self):
        self.evaluator.procedures_discharge = ["Drainage of intra-abdominal space"]
        self.evaluator.answers["Treatment"] = "Drain the pelvic abscess"
        self.evaluator.score_treatment()

        self.assertEqual(
            self.evaluator.answers["Treatment Requested"],
            {
                "Colonoscopy": False,
                "Antibiotics": False,
                "Support": False,
                "Drainage": True,
                "Colectomy": False,
            },
        )

        self.assertEqual(
            self.evaluator.answers["Treatment Required"],
            {
                "Colonoscopy": True,
                "Antibiotics": True,
                "Support": True,
                "Drainage": True,
                "Colectomy": False,
            },
        )

    def test_score_treatment_drainage_discharge_wrong_organ(self):
        self.evaluator.procedures_discharge = ["Drainage of chest"]
        self.evaluator.answers["Treatment"] = "Drain the pelvic abscess"
        self.evaluator.score_treatment()

        self.assertEqual(
            self.evaluator.answers["Treatment Requested"],
            {
                "Colonoscopy": False,
                "Antibiotics": False,
                "Support": False,
                "Drainage": True,
                "Colectomy": False,
            },
        )

        self.assertEqual(
            self.evaluator.answers["Treatment Required"],
            {
                "Colonoscopy": True,
                "Antibiotics": True,
                "Support": True,
                "Drainage": False,
                "Colectomy": False,
            },
        )

    def test_score_treatment_drainage_requested_not_required(self):
        self.evaluator.answers["Treatment"] = "Sigmoid colon drainage"
        self.evaluator.score_treatment()

        self.assertEqual(
            self.evaluator.answers["Treatment Requested"],
            {
                "Colonoscopy": False,
                "Antibiotics": False,
                "Support": False,
                "Drainage": True,
                "Colectomy": False,
            },
        )

        self.assertEqual(
            self.evaluator.answers["Treatment Required"],
            {
                "Colonoscopy": True,
                "Antibiotics": True,
                "Support": True,
                "Drainage": False,
                "Colectomy": False,
            },
        )

    def test_score_treatment_drainage_requested_not_required_2(self):
        self.evaluator.answers["Treatment"] = "Drain the abscess"
        self.evaluator.score_treatment()

        self.assertEqual(
            self.evaluator.answers["Treatment Requested"],
            {
                "Colonoscopy": False,
                "Antibiotics": False,
                "Support": False,
                "Drainage": True,
                "Colectomy": False,
            },
        )

        self.assertEqual(
            self.evaluator.answers["Treatment Required"],
            {
                "Colonoscopy": True,
                "Antibiotics": True,
                "Support": True,
                "Drainage": False,
                "Colectomy": False,
            },
        )

    def test_score_treatment_colectomy_icd10(self):
        self.evaluator.answers["Treatment"] = "Colectomy"
        self.evaluator.procedures_icd10 = ["0DBM8ZZ"]
        self.evaluator.score_treatment()

        self.assertEqual(
            self.evaluator.answers["Treatment Requested"],
            {
                "Colonoscopy": False,
                "Antibiotics": False,
                "Support": False,
                "Drainage": False,
                "Colectomy": True,
            },
        )

        self.assertEqual(
            self.evaluator.answers["Treatment Required"],
            {
                "Colonoscopy": True,
                "Antibiotics": True,
                "Support": True,
                "Drainage": False,
                "Colectomy": True,
            },
        )

    def test_score_treatment_colectomy_icd9(self):
        self.evaluator.answers["Treatment"] = "Colectomy"
        self.evaluator.procedures_icd9 = [4542]
        self.evaluator.score_treatment()

        self.assertEqual(
            self.evaluator.answers["Treatment Requested"],
            {
                "Colonoscopy": False,
                "Antibiotics": False,
                "Support": False,
                "Drainage": False,
                "Colectomy": True,
            },
        )

        self.assertEqual(
            self.evaluator.answers["Treatment Required"],
            {
                "Colonoscopy": True,
                "Antibiotics": True,
                "Support": True,
                "Drainage": False,
                "Colectomy": True,
            },
        )

    def test_score_treatment_colectomy_discharge(self):
        self.evaluator.answers["Treatment"] = "Colectomy"
        self.evaluator.procedures_discharge = ["COLECTOMY"]
        self.evaluator.score_treatment()

        self.assertEqual(
            self.evaluator.answers["Treatment Requested"],
            {
                "Colonoscopy": False,
                "Antibiotics": False,
                "Support": False,
                "Drainage": False,
                "Colectomy": True,
            },
        )

        self.assertEqual(
            self.evaluator.answers["Treatment Required"],
            {
                "Colonoscopy": True,
                "Antibiotics": True,
                "Support": True,
                "Drainage": False,
                "Colectomy": True,
            },
        )

    def test_score_treatment_colectomy_requested_not_required(self):
        self.evaluator.answers["Treatment"] = "Colectomy"
        self.evaluator.score_treatment()

        self.assertEqual(
            self.evaluator.answers["Treatment Requested"],
            {
                "Colonoscopy": False,
                "Antibiotics": False,
                "Support": False,
                "Drainage": False,
                "Colectomy": True,
            },
        )

        self.assertEqual(
            self.evaluator.answers["Treatment Required"],
            {
                "Colonoscopy": True,
                "Antibiotics": True,
                "Support": True,
                "Drainage": False,
                "Colectomy": False,
            },
        )

    def test_score_treatment_cholecystitis_cholecystectomy_alternate_keywords(self):
        self.evaluator.answers["Treatment"] = "Surgical removal parts of the colon."
        self.evaluator.score_treatment()

        self.assertEqual(
            self.evaluator.answers["Treatment Requested"],
            {
                "Colonoscopy": False,
                "Antibiotics": False,
                "Support": False,
                "Drainage": False,
                "Colectomy": True,
            },
        )

        self.assertEqual(
            self.evaluator.answers["Treatment Required"],
            {
                "Colonoscopy": True,
                "Antibiotics": True,
                "Support": True,
                "Drainage": False,
                "Colectomy": False,
            },
        )

    def test_score_treatment_all(self):
        self.evaluator.answers[
            "Treatment"
        ] = "Colonoscopy, Antibiotics, Drain the abdomen, Colectomy, Monitoring, Fluids, Analgesia"
        self.evaluator.procedures_icd10 = []
        self.evaluator.procedures_icd9 = [4542, 5491]
        self.evaluator.score_treatment()

        self.assertEqual(
            self.evaluator.answers["Treatment Requested"],
            {
                "Colonoscopy": True,
                "Antibiotics": True,
                "Support": True,
                "Drainage": True,
                "Colectomy": True,
            },
        )

        self.assertEqual(
            self.evaluator.answers["Treatment Required"],
            {
                "Colonoscopy": True,
                "Antibiotics": True,
                "Support": True,
                "Drainage": True,
                "Colectomy": True,
            },
        )

    @unittest.skip("Logic is too complicated")
    def test_score_treatment_colonoscopy_time(self):
        self.evaluator.answers["Treatment"] = "Colonoscopy after treatment"
        self.evaluator.score_treatment()

    @unittest.skip("Logic is too complicated")
    def test_check_colonoscopy_time_order(self):
        self.evaluator.answers["Treatment"] = "Colonoscopy"
        self.assertTrue(self.evaluator.check_colonoscopy_time_order())

    @unittest.skip("Logic is too complicated")
    def test_check_colonoscopy_time_order_1(self):
        self.evaluator.answers["Treatment"] = "Colonoscopy after treatment"
        self.assertTrue(self.evaluator.check_colonoscopy_time_order())

    @unittest.skip("Logic is too complicated")
    def test_check_colonoscopy_time_order_2(self):
        self.evaluator.answers["Treatment"] = "After treatment do a Colonoscopy"
        self.assertTrue(self.evaluator.check_colonoscopy_time_order())

    @unittest.skip("Logic is too complicated")
    def test_check_colonoscopy_time_order_3(self):
        self.evaluator.answers["Treatment"] = "Cure the patient then do a colonoscopy"
        self.assertTrue(self.evaluator.check_colonoscopy_time_order())


if __name__ == "__main__":
    unittest.main()
