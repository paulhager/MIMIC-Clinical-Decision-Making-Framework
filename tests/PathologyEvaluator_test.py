import unittest
from evaluators.appendicitis_evaluator import AppendicitisEvaluator
from agents.AgentAction import AgentAction
from agents.DiagnosisWorkflowParser import InvalidActionError


class TestPathologyEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = AppendicitisEvaluator()
        self.maxDiff = None
        self.base_app_reference = (
            "Acute Appendicitis",
            ["Unspecified acute appendicitis without abscess"],
            [],
            [],
            [],
            [],
        )

    ########################
    # PHYSICAL EXAMINATION #
    ########################

    def test_score_physical_examination_first_action(self):
        action = AgentAction(
            tool="Physical Examination", tool_input="", log="", custom_parsings=0
        )

        self.evaluator.score_physical_examination(action, indx=0)

        self.assertEqual(self.evaluator.scores["Physical Examination"], 1)
        self.assertEqual(self.evaluator.scores["Late Physical Examination"], 1)

    def test_score_physical_examination_not_first_action(self):
        action = AgentAction(
            tool="Physical Examination", tool_input="", log="", custom_parsings=0
        )

        self.evaluator.score_physical_examination(action, indx=1)

        self.assertEqual(self.evaluator.scores["Physical Examination"], 0)
        self.assertEqual(self.evaluator.scores["Late Physical Examination"], 1)

    def test_evaluator_physical_examination(self):
        steps = []
        action = AgentAction(
            tool="Physical Examination", tool_input="", log="", custom_parsings=0
        )
        observation = (
            "Patient presents with RLQ pain, rebound tenderness, and guarding."
        )
        steps.append((action, observation))

        result = {
            "input": "",
            "output": "",
            "intermediate_steps": steps,
        }
        eval = self.evaluator._evaluate_agent_trajectory(
            prediction=result["output"],
            input=result["input"],
            reference=self.base_app_reference,
            agent_trajectory=result["intermediate_steps"],
        )

        self.assertEqual(eval["scores"]["Physical Examination"], 1)
        self.assertEqual(eval["scores"]["Late Physical Examination"], 1)

    ##############
    # LABORATORY #
    ##############

    def test_score_laboratory_tests(self):
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

    def test_score_laboratory_tests_multiple_required(self):
        tests_ordered = [51301, 50889]  # White Blood Cells, CRP
        action = AgentAction(
            tool="Laboratory Tests",
            tool_input={"action_input": tests_ordered},
            log="",
            custom_parsings=0,
        )

        self.evaluator.score_laboratory_tests(action)

        self.assertEqual(self.evaluator.scores["Laboratory Tests"], 1)
        self.assertEqual(
            self.evaluator.answers["Correct Laboratory Tests"],
            {"Inflammation": [51301, 50889]},
        )
        self.assertEqual(self.evaluator.answers["Unnecessary Laboratory Tests"], [])

    def test_score_laboratory_tests_multiple_rounds(self):
        tests_ordered = [51301]  # White Blood Cells
        action = AgentAction(
            tool="Laboratory Tests",
            tool_input={"action_input": tests_ordered},
            log="",
            custom_parsings=0,
        )

        self.evaluator.score_laboratory_tests(action)
        self.evaluator.score_laboratory_tests(action)

        self.assertEqual(self.evaluator.scores["Laboratory Tests"], 1)
        self.assertEqual(
            self.evaluator.answers["Correct Laboratory Tests"],
            {"Inflammation": [51301, 51301]},
        )
        self.assertEqual(self.evaluator.answers["Unnecessary Laboratory Tests"], [])

    def test_score_laboratory_tests_neutral(self):
        tests_ordered = [51279]  # Red Blood Cells
        action = AgentAction(
            tool="Laboratory Tests",
            tool_input={"action_input": tests_ordered},
            log="",
            custom_parsings=0,
        )

        self.evaluator.score_laboratory_tests(action)

        self.assertEqual(self.evaluator.scores["Laboratory Tests"], 0)
        self.assertEqual(
            self.evaluator.answers["Correct Laboratory Tests"], {"Inflammation": []}
        )
        self.assertEqual(self.evaluator.answers["Unnecessary Laboratory Tests"], [])

    def test_score_laboratory_tests_unnecessary(self):
        tests_ordered = ["Fake Test 1", "Fake Test 2", "Fake Test 3"]
        action = AgentAction(
            tool="Laboratory Tests",
            tool_input={"action_input": tests_ordered},
            log="",
            custom_parsings=0,
        )

        self.evaluator.score_laboratory_tests(action)

        self.assertEqual(self.evaluator.scores["Laboratory Tests"], 0)
        self.assertEqual(
            self.evaluator.answers["Correct Laboratory Tests"], {"Inflammation": []}
        )
        self.assertEqual(
            self.evaluator.answers["Unnecessary Laboratory Tests"], tests_ordered
        )

    def test_score_laboratory_tests_correct_and_unnecessary(self):
        tests_ordered = [51301]  # White Blood Cells
        tests_ordered.append("Fake Test 1")
        tests_ordered.append("Fake Test 2")
        tests_ordered.append("Fake Test 3")
        action = AgentAction(
            tool="Laboratory Tests",
            tool_input={"action_input": tests_ordered},
            log="",
            custom_parsings=0,
        )

        self.evaluator.score_laboratory_tests(action)

        self.assertEqual(self.evaluator.scores["Laboratory Tests"], 1)
        self.assertEqual(
            self.evaluator.answers["Correct Laboratory Tests"],
            {"Inflammation": [51301]},
        )
        self.assertEqual(
            self.evaluator.answers["Unnecessary Laboratory Tests"],
            ["Fake Test 1", "Fake Test 2", "Fake Test 3"],
        )

    def test_evaluator_laboratory_tests(self):
        steps = []
        tests_ordered = [51301]  # White Blood Cells
        action = AgentAction(
            tool="Laboratory Tests",
            tool_input={"action_input": tests_ordered},
            log="",
            custom_parsings=0,
        )
        observation = ""
        steps.append((action, observation))

        result = {
            "input": "",
            "output": "",
            "intermediate_steps": steps,
        }
        eval = self.evaluator._evaluate_agent_trajectory(
            prediction=result["output"],
            input=result["input"],
            reference=self.base_app_reference,
            agent_trajectory=result["intermediate_steps"],
        )

        self.assertEqual(eval["scores"]["Laboratory Tests"], 1)
        self.assertEqual(
            eval["answers"]["Correct Laboratory Tests"],
            {"Inflammation": [51301]},
        )
        self.assertEqual(eval["answers"]["Unnecessary Laboratory Tests"], [])

    ###########
    # IMAGING #
    ###########

    def test_evaluator_imaging(self):
        steps = []
        imaging_dict = {"region": "Abdomen", "modality": "CT"}
        action = AgentAction(
            tool="Imaging",
            tool_input={"action_input": imaging_dict},
            log="",
            custom_parsings=0,
        )
        observation = ""
        steps.append((action, observation))

        result = {
            "input": "",
            "output": "",
            "intermediate_steps": steps,
        }
        eval = self.evaluator._evaluate_agent_trajectory(
            prediction=result["output"],
            input=result["input"],
            reference=self.base_app_reference,
            agent_trajectory=result["intermediate_steps"],
        )

        self.assertEqual(eval["scores"]["Imaging"], 1)
        self.assertEqual(eval["answers"]["Correct Imaging"], [imaging_dict])
        self.assertEqual(eval["answers"]["Unnecessary Imaging"], [])

    #############
    # DIAGNOSIS #
    #############

    def test_parse_and_score_diagnosis(self):
        prediction = "Final Diagnosis: Acute Appendicitis"

        self.evaluator.parse_diagnosis(prediction)
        self.evaluator.score_diagnosis()

        self.assertEqual(self.evaluator.scores["Diagnosis"], 1)
        self.assertEqual(self.evaluator.answers["Diagnosis"], "Acute Appendicitis")
        self.assertEqual(self.evaluator.scores["Diagnosis Parsing"], 0)

    def test_parse_and_score_diagnosis_multiline_1(self):
        prediction = "Final Diagnosis:\nAcute Appendicitis"

        self.evaluator.parse_diagnosis(prediction)
        self.evaluator.score_diagnosis()

        self.assertEqual(self.evaluator.scores["Diagnosis"], 1)
        self.assertEqual(self.evaluator.answers["Diagnosis"], "Acute Appendicitis")
        self.assertEqual(self.evaluator.scores["Diagnosis Parsing"], 0)

    def test_parse_and_score_diagnosis_multiline_2(self):
        prediction = "Final Diagnosis:\nAcute Appendicitis\n\nTreatment:\nAppendectomy"

        self.evaluator.parse_diagnosis(prediction)
        self.evaluator.score_diagnosis()

        self.assertEqual(self.evaluator.scores["Diagnosis"], 1)
        self.assertEqual(self.evaluator.answers["Diagnosis"], "Acute Appendicitis")
        self.assertEqual(self.evaluator.scores["Diagnosis Parsing"], 0)

    def test_score_diagnosis(self):
        self.evaluator.answers["Diagnosis"] = "Acute Appendicitis"

        self.evaluator.score_diagnosis()

        self.assertEqual(self.evaluator.scores["Diagnosis"], 1)

    def test_score_diagnosis_short(self):
        self.evaluator.answers["Diagnosis"] = "Appendicitis"

        self.evaluator.score_diagnosis()

        self.assertEqual(self.evaluator.scores["Diagnosis"], 1)

    def test_score_diagnosis_lower_case(self):
        self.evaluator.answers["Diagnosis"] = "acute appendicitis"

        self.evaluator.score_diagnosis()

        self.assertEqual(self.evaluator.scores["Diagnosis"], 1)

    def test_score_diagnosis_short_lower_case(self):
        self.evaluator.answers["Diagnosis"] = "appendicitis"

        self.evaluator.score_diagnosis()

        self.assertEqual(self.evaluator.scores["Diagnosis"], 1)

    def test_score_diagnosis_spelling_mistake(self):
        self.evaluator.answers["Diagnosis"] = "apendicitis"

        self.evaluator.score_diagnosis()

        self.assertEqual(self.evaluator.scores["Diagnosis"], 1)

    def test_score_diagnosis_incorrect(self):
        self.evaluator.answers["Diagnosis"] = "Acute Pancreatitis"

        self.evaluator.score_diagnosis()

        self.assertEqual(self.evaluator.scores["Diagnosis"], 0)

    def test_score_diagnosis_negation_1(self):
        self.evaluator.answers[
            "Diagnosis"
        ] = "The patient does not have Acute Appendicitis"

        self.evaluator.score_diagnosis()

        self.assertEqual(self.evaluator.scores["Diagnosis"], 0)

    def test_score_diagnosis_negation_2(self):
        self.evaluator.answers["Diagnosis"] = "No Acute Appendicitis"

        self.evaluator.score_diagnosis()

        self.assertEqual(self.evaluator.scores["Diagnosis"], 0)

    def test_score_diagnosis_negation_3(self):
        self.evaluator.answers[
            "Diagnosis"
        ] = "There is no evidence of acute appendicitis."

        self.evaluator.score_diagnosis()

        self.assertEqual(self.evaluator.scores["Diagnosis"], 0)

    def test_score_diagnosis_negation_4(self):
        self.evaluator.answers["Diagnosis"] = "No signs of acute appendicitis."

        self.evaluator.score_diagnosis()

        self.assertEqual(self.evaluator.scores["Diagnosis"], 0)

    def test_score_diagnosis_negation_5(self):
        self.evaluator.answers[
            "Diagnosis"
        ] = "Absence of typical signs and symptoms of appendicitis."

        self.evaluator.score_diagnosis()

        self.assertEqual(self.evaluator.scores["Diagnosis"], 0)

    def test_parse_diagnosis(self):
        prediction = "Final Diagnosis: Acute Appendicitis"

        self.evaluator.parse_diagnosis(prediction)

        self.assertEqual(self.evaluator.answers["Diagnosis"], "Acute Appendicitis")
        self.assertEqual(self.evaluator.scores["Diagnosis Parsing"], 0)

    def test_parse_diagnosis_empty(self):
        prediction = "Final Diagnosis:"

        self.evaluator.parse_diagnosis(prediction)

        self.assertEqual(self.evaluator.answers["Diagnosis"], "")
        self.assertEqual(self.evaluator.scores["Diagnosis Parsing"], 0)

    def test_parse_diagnosis_lower_case(self):
        prediction = "final diagnosis: Acute Appendicitis"

        self.evaluator.parse_diagnosis(prediction)

        self.assertEqual(self.evaluator.answers["Diagnosis"], "Acute Appendicitis")
        self.assertEqual(self.evaluator.scores["Diagnosis Parsing"], 0)

    def test_parse_diagnosis_no_final_before_diagnosis(self):
        prediction = "Diagnosis: Acute Appendicitis"

        self.evaluator.parse_diagnosis(prediction)

        self.assertEqual(self.evaluator.answers["Diagnosis"], "Acute Appendicitis")
        self.assertEqual(self.evaluator.scores["Diagnosis Parsing"], 1)

    def test_parse_diagnosis_incorrect_format(self):
        prediction = "The diagnosis is Acute Appendicitis"

        self.evaluator.parse_diagnosis(prediction)

        self.assertEqual(self.evaluator.answers["Diagnosis"], "")
        self.assertEqual(self.evaluator.scores["Diagnosis Parsing"], 0)

    def test_parse_diagnosis_double_diagnosis(self):
        prediction = "Diagnosis: Abdominal pain\nTreatment Plan: Management of pain\nFinal Diagnosis: Appendicitis\nFinal Treatment:Appendectomy"

        self.evaluator.parse_diagnosis(prediction)

        self.assertEqual(self.evaluator.answers["Diagnosis"], "Abdominal pain")
        self.assertEqual(self.evaluator.scores["Diagnosis Parsing"], 1)

    def test_parse_diagnosis_long_treatment_string(self):
        prediction = "Final Diagnosis: Appendicitis\nBest Treatment Plan: Appendectomy"

        self.evaluator.parse_diagnosis(prediction)

        self.assertEqual(self.evaluator.answers["Diagnosis"], "Appendicitis")
        self.assertEqual(self.evaluator.scores["Diagnosis Parsing"], 0)

    def test_parse_diagnosis_one_line_with_period(self):
        prediction = "Final Diagnosis: Appendicitis. Best Treatment Plan: Appendectomy"

        self.evaluator.parse_diagnosis(prediction)

        self.assertEqual(self.evaluator.answers["Diagnosis"], "Appendicitis")
        self.assertEqual(self.evaluator.scores["Diagnosis Parsing"], 0)

    def test_parse_diagnosis_multiple_diagnoses_vs(self):
        prediction = "Final Diagnosis: Pancreatitis vs Appendicitis"

        self.evaluator.parse_diagnosis(prediction)

        self.assertEqual(self.evaluator.answers["Diagnosis"], "Pancreatitis")
        self.assertEqual(self.evaluator.scores["Diagnosis Parsing"], 1)

    def test_parse_diagnosis_multiple_diagnoses_vs_period(self):
        prediction = "Final Diagnosis: Pancreatitis vs. Appendicitis"

        self.evaluator.parse_diagnosis(prediction)

        self.assertEqual(self.evaluator.answers["Diagnosis"], "Pancreatitis")
        self.assertEqual(self.evaluator.scores["Diagnosis Parsing"], 1)

    def test_parse_diagnosis_multiple_diagnoses_newline(self):
        prediction = "Final Diagnosis: Pancreatitis\nAppendicitis"

        self.evaluator.parse_diagnosis(prediction)

        self.assertEqual(self.evaluator.answers["Diagnosis"], "Pancreatitis")
        self.assertEqual(self.evaluator.scores["Diagnosis Parsing"], 1)

    def test_parse_diagnosis_multiple_diagnoses_double_newline(self):
        prediction = "Final Diagnosis:\nPancreatitis\nAppendicitis"

        self.evaluator.parse_diagnosis(prediction)

        self.assertEqual(self.evaluator.answers["Diagnosis"], "Pancreatitis")
        self.assertEqual(self.evaluator.scores["Diagnosis Parsing"], 1)

    def test_parse_diagnosis_multiple_diagnoses_or(self):
        prediction = "Final Diagnosis: Pancreatitis or Appendicitis"

        self.evaluator.parse_diagnosis(prediction)

        self.assertEqual(self.evaluator.answers["Diagnosis"], "Pancreatitis")
        self.assertEqual(self.evaluator.scores["Diagnosis Parsing"], 1)

    def test_parse_diagnosis_multiple_diagnoses_and(self):
        prediction = "Final Diagnosis: Pancreatitis and Appendicitis"

        self.evaluator.parse_diagnosis(prediction)

        self.assertEqual(self.evaluator.answers["Diagnosis"], "Pancreatitis")
        self.assertEqual(self.evaluator.scores["Diagnosis Parsing"], 1)

    def test_parse_diagnosis_multiple_diagnoses_comma(self):
        prediction = "Final Diagnosis: Pancreatitis, Appendicitis"

        self.evaluator.parse_diagnosis(prediction)

        self.assertEqual(self.evaluator.answers["Diagnosis"], "Pancreatitis")
        self.assertEqual(self.evaluator.scores["Diagnosis Parsing"], 1)

    def test_parse_diagnosis_multiple_diagnoses_three_plus(self):
        prediction = "Final Diagnosis: Pancreatitis vs Appendicitis vs Cholecystitis"

        self.evaluator.parse_diagnosis(prediction)

        self.assertEqual(self.evaluator.answers["Diagnosis"], "Pancreatitis")
        self.assertEqual(self.evaluator.scores["Diagnosis Parsing"], 1)

    def test_parse_diagnosis_numbering(self):
        prediction = "Final Diagnosis:\n1.Pancreatitis\n2.Appendicitis"

        self.evaluator.parse_diagnosis(prediction)

        self.assertEqual(self.evaluator.answers["Diagnosis"], "Pancreatitis")
        self.assertEqual(self.evaluator.scores["Diagnosis Parsing"], 1)

    def test_parse_diagnosis_rationale(self):
        predict = """Final Diagnosis: Acute Kidney Injury (AKI) with Septic Shock, likely caused by Pyelonephritis.

Rationale:

1. The patient's history of recurrent UTIs and pyelonephritis, along with the presence of hematuria and proteinuria in the urinalysis, suggest a possible underlying kidney infection.
2. The patient's recent fever, nausea, vomiting, diarrhea, and abdominal pain are consistent with septic shock due to pyelonephritis.
3. The patient's lab results show elevated white blood cell count, neutrophilia, and elevated serum creatinine, which support the diagnosis of AKI.
4. The patient's urine output, specific gravity, and positive nitrite test suggest that the patient may have a urinary tract infection.
5. The imaging studies did not reveal any obvious obstruction or damage to the transplanted kidney, which supports the diagnosis of pyelonephritis rather than acute rejection.
6. The patient's history of previous episodes of pyelonephritis and the presence of a ureteral stent suggest that the patient may be more prone to developing pyelonephritis."""
        self.evaluator.parse_diagnosis(predict)

        self.assertEqual(
            self.evaluator.answers["Diagnosis"],
            "Acute Kidney Injury (AKI) with Septic Shock",
        )
        self.assertEqual(self.evaluator.scores["Diagnosis Parsing"], 1)

    def test_parse_diagnosis_based_on(self):
        predict = """Final Diagnosis: Based on the patient's symptoms, physical examination, laboratory results, imaging findings, and medical history, the most likely diagnosis is:

Acute Appendicitis

The patient's symptoms of sudden-onset abdominal pain, nausea, vomiting, fever, and chills, along with the physical examination findings of tenderness in the right lower quadrant, guarding, and positive Rovsing's sign, are consistent with acute appendicitis. The laboratory results show a mild leukocytosis and elevated neutrophils, which support the diagnosis. The CT scan findings of an enlarged, fluid-filled appendix with wall thickening, hyperemia, and surrounding inflammation are also consistent with acute appendicitis. The patient's history of depression increases the likelihood of a surgical emergency. Therefore, the patient should be treated urgently with broad-spectrum antibiotics and surgical intervention to remove the inflamed appendix."""
        self.evaluator.parse_diagnosis(predict)

        self.assertEqual(self.evaluator.answers["Diagnosis"], "Acute Appendicitis")
        self.assertEqual(self.evaluator.scores["Diagnosis Parsing"], 1)

    def test_parse_diagnosis_based_on_2(self):
        predict = """Final Diagnosis: Based on the provided information, the patient's final diagnosis is most likely a retroperitoneal sarcoma. This conclusion is based on the patient's symptoms, physical examination, laboratory results, imaging studies (CT and MRI), and medical history."""

        self.evaluator.parse_diagnosis(predict)

        self.assertEqual(
            self.evaluator.answers["Diagnosis"], "most likely a retroperitoneal sarcoma"
        )
        self.assertEqual(self.evaluator.scores["Diagnosis Parsing"], 1)

    def test_parse_diagnosis_based_on_mid_sentence(self):
        predict = """Final Diagnosis: Based on the patient's symptoms and laboratory results, the most likely diagnosis is acute appendicitis. The patient's sudden onset of severe pain in the lower right abdomen, accompanied by nausea, vomiting, and loss of appetite, are classic symptoms of appendicitis. The physical examination findings of tenderness and guarding in the right lower quadrant of the abdomen, along with the presence of a positive Rovsings sign and Obturator sign, also support this diagnosis. Additionally, the patient's elevated white blood cell count and neutrophilia suggest an inflammatory response, consistent with appendicitis. The absence of blood in the patient's emesis and the presence of a small bowel obstruction also support this diagnosis. The patient's history of previous cancer and recent severe back pain may also be relevant to the patient's overall health and potential treatment plan, but they do not change the most likely diagnosis of acute appendicitis."""
        self.evaluator.parse_diagnosis(predict)

        self.assertEqual(self.evaluator.answers["Diagnosis"], "acute appendicitis")
        self.assertEqual(self.evaluator.scores["Diagnosis Parsing"], 1)

    def test_parse_diagnosis_explanation(self):
        prediction = """Final Diagnosis: Acute Appendicitis. 

The patient's symptoms of sudden-onset right lower quadrant pain, anorexia, nausea, vomiting, and fever, along with physical examination findings of tenderness and guarding in the right lower abdomen, suggest acute appendicitis. This suspicion is supported by the laboratory results showing a slight increase in white blood cell count and neutrophils. The CT scan also shows evidence of appendiceal inflammation and enlarged lymph nodes near the ascending colon. Additionally, the absence of other significant medical conditions in the patient's past medical history and family history supports this diagnosis. Therefore, acute appendicitis is the most likely diagnosis."""
        self.evaluator.parse_diagnosis(prediction)

        self.assertEqual(self.evaluator.answers["Diagnosis"], "Acute Appendicitis")
        self.assertEqual(self.evaluator.scores["Diagnosis Parsing"], 1)

    def test_parse_diagnosis_explanation_multi(self):
        prediction = """Final Diagnosis: Acute Appendicitis. 

The patient's symptoms of sudden-onset lower abdominal pain, radiation to the groin, nausea, and vomiting, along with the physical examination findings of tenderness in the suprapubic region and right groin, are consistent with acute appendicitis. The absence of fever, chills, or rigors suggests an early stage of appendicitis. The patient's history of premature birth and cryptorchidism increases the likelihood of an underlying genetic disorder, but it is unlikely to be related to the current presentation.

The laboratory results show a slight elevation in white blood cell count, neutrophils, and monocytes, which supports the diagnosis of appendicitis. The CT scan shows an enlarged appendix with periappendiceal stranding, indicating inflammation. Additionally, the presence of a hypodense structure in the right inguinal canal, extending into the right pelvis, is consistent with an undescended testicle.

Therefore, the final diagnosis is acute appendicitis, and the patient should undergo immediate surgical intervention to remove the inflamed appendix."""
        self.evaluator.parse_diagnosis(prediction)

        self.assertEqual(self.evaluator.answers["Diagnosis"], "Acute Appendicitis")
        self.assertEqual(self.evaluator.scores["Diagnosis Parsing"], 1)

    def test_parse_diagnosis_star_list(self):
        prediction = """Final Diagnosis: According to the provided information, the patient has terminal ileitis with a normal appendix, periappendiceal abscess, and pelvic abscess. Additionally, there is evidence of prior surgery (appendectomy) and a history of smoking. 

Given these findings, the most likely diagnosis for this patient would be:

* Appendiceal cancer with periappendiceal abscess and pelvic abscess.
* Chronic smoker"""

        self.evaluator.parse_diagnosis(prediction)

        self.assertEqual(
            self.evaluator.answers["Diagnosis"],
            "Appendiceal cancer with periappendiceal abscess",
        )
        self.assertEqual(self.evaluator.scores["Diagnosis Parsing"], 1)

    def test_parse_diagnosis_delete_unwanted_section(self):
        prediction = """Final Diagnosis: Metastatic ovarian cancer with liver metastases, stable since previous imaging.

Additional notes:

* The patient has a history of ovarian cancer with metastasis to the liver, diaphragm, and cul-de-sac.
* The current CT scan shows multiple lesions in the liver, including a 1.5 x 1.0 cm bilobed cystic lesion in the right liver dome and two lesions along the diaphragmatic contour of the left lobe of the liver.
* There is also a sub-cm hypodensity in the lower pole of the right kidney, likely a cyst.
* The patient has been treated with chemotherapy, including carboplatin and taxol, and has had surgery, including TAH/BSO, rectosigmoid resection with primary anastomosis, complete omentectomy, radical ureterolysis, resection of umbilicus, argon beam ablation of metastatic tumor deposits, excision of diaphragmatic tumor nodules, and argon beam cauterization of the right and left hemidiaphragms.
* The patient has chronic hepatitis C and liver biopsy demonstrated stage 1 fibrosis and grade 1 inflammation."""

        self.evaluator.parse_diagnosis(prediction)

        self.assertEqual(
            self.evaluator.answers["Diagnosis"],
            "Metastatic ovarian cancer with liver metastases",
        )
        self.assertEqual(self.evaluator.scores["Diagnosis Parsing"], 1)

    def test_parse_diagnosis_diagnosis_in_line_late(self):
        prediction = """Final Diagnosis: Based on the patient's history and physical examination, it appears that she has been experiencing persistent right lower quadrant abdominal pain, which has been worsening over the past four days. Her lab results show a mild elevation in her white blood cell count and a low grade fever. Given these symptoms and lab results, it is likely that the patient has acute appendicitis. However, because the patient has a history of depression and anxiety, it is also possible that her symptoms could be related to a somatic delusional disorder or a factitious disorder. Therefore, it would be important to consider both possibilities and rule out other potential causes of her symptoms before making a definitive diagnosis. 

In light of this, I recommend that the patient undergo an emergency appendectomy to treat her acute appendicitis. Additionally, I suggest that the patient follow up with a psychiatrist to address any underlying mental health issues that may be contributing to her symptoms. 

As for further workup, I recommend that the patient undergo a CT scan of the abdomen and pelvis with IV contrast to better evaluate the appendix and surrounding tissues. If the CT scan confirms the diagnosis of acute appendicitis, then an emergency appendectomy should be performed. If the CT scan does not reveal any signs of appendicitis, then further investigation into the patient's mental health status should be pursued. 

In summary, my final diagnosis is acute appendicitis vs. somatic delusional disorder vs. factitious disorder. I recommend an emergency appendectomy and a follow-up with a psychiatrist to address any underlying mental health issues."""

        self.evaluator.parse_diagnosis(prediction)

        self.assertEqual(self.evaluator.answers["Diagnosis"], "acute appendicitis")
        self.assertEqual(self.evaluator.scores["Diagnosis Parsing"], 1)

    def test_parse_diagnosis_llama_2_chat_1(self):
        prediction = """Final Diagnosis: Based on the patient's history, physical examination, laboratory results, imaging findings, and the absence of any other explanatory conditions, the most likely diagnosis is:

Diabetic ketoacidosis (DKA) with suspected urinary tract infection (UTI)

The patient's symptoms of weakness, malaise, polydipsia, polyuria, and weight loss are consistent with DKA. The laboratory results show elevated blood glucose, ketones in the urine, metabolic acidosis, and renal impairment, which support the diagnosis of DKA. The presence of pyelonephritis (kidney infection) cannot be ruled out with certainty, but it is less likely given the absence of fever, chills, or flank pain. However, the possibility of a UTI should be considered and investigated further.

Other possible diagnoses that were considered include:

* Acute kidney injury (AKI): The patient's elevated creatinine level and reduced estimated glomerular filtration rate (eGFR) suggest AKI. However, the absence of any obvious precipitating factors, such as hypovolemia, sepsis, or nephrotoxins, makes AKI less likely.
* Chronic kidney disease (CKD): The patient's history of kidney donation and hypertension increases the likelihood of CKD. However, the patient's relatively preserved kidney function and absence of proteinuria do not support this diagnosis.
* Gastrointestinal pathology: The patient's abdominal discomfort and diarrhea raise the possibility of gastrointestinal pathology, such as inflammatory bowel disease or irritable bowel syndrome. However, the absence of any other gastrointestinal symptoms, such as nausea, vomiting, or rectal bleeding, makes this diagnosis less likely.
* Pancreatitis: The patient's elevated amylase and lipase levels suggest pancreatitis. However, the absence of abdominal pain, nausea, or vomiting makes this diagnosis less likely.

In conclusion, the patient's presentation and laboratory results are most consistent with diabetic ketoacidosis (DKA) with suspected urinary tract infection (UTI). Further investigation and management should focus on addressing these conditions and ruling out other potential causes of the patient's symptoms."""

        self.evaluator.parse_diagnosis(prediction)

        self.assertEqual(
            self.evaluator.answers["Diagnosis"],
            "Diabetic ketoacidosis (DKA) with suspected urinary tract infection (UTI)",
        )
        self.assertEqual(self.evaluator.scores["Diagnosis Parsing"], 1)

    def test_parse_diagnosis_llama_2_chat_2(self):
        prediction = """Final Diagnosis: Based on the patient's history, physical examination, laboratory results, imaging studies, and other findings, the final diagnosis is:

1. Chronic Myeloid Leukemia (CML) - This diagnosis is supported by the patient's history of CML, stable on Gleevec, and the presence of splenomegaly on CT scan.
2. Ischemic Cardiomyopathy - This diagnosis is supported by the patient's history of ischemic cardiomyopathy with an EF of 35%, and the presence of inferior akinesis, 1+ MR, and LMCA, LAD, and LCX on echocardiography.
3. Type 2 Diabetes Mellitus - This diagnosis is supported by the patient's history of type 2 diabetes, diet-controlled.
4. Atrial Fibrillation/Flutter - This diagnosis is supported by the patient's history of atrial fibrillation/flutter and the use of Coumadin and an ICD/PPM.
5. Chronic Obstructive Pulmonary Disease (COPD) - This diagnosis is supported by the patient's history of COPD and the presence of bibasilar atelectasis on CT scan.
6. Gastroesophageal Reflux Disease (GERD) - This diagnosis is supported by the patient's history of GERD and the presence of esophagitis on endoscopy.
7. Mesenteric Ischemia - This diagnosis is supported by the patient's history of mesenteric ischemia and the presence of a partially imaged lead within the right ventricle on CT scan.
8. Osteoporosis - This diagnosis is supported by the patient's history of osteoporosis and the presence of degenerative changes in the lumbar spine on CT scan.
9. Psoriasis - This diagnosis is supported by the patient's history of psoriasis and the presence of pruritus.
10. Hypothyroidism - This diagnosis is supported by the patient's history of hypothyroidism.

Other possible diagnoses that cannot be ruled out based on the available information include:

* Gallbladder disease - The patient's history of cholelithiasis and cholecystitis suggests this diagnosis.
* Pancreatitis - The patient's history of pancreatitis and the presence of pancreatic enzymes in the blood suggest this diagnosis.
* Liver disease - The patient's history of liver disease and the presence of a hemangioma in the liver suggest this diagnosis.
* Adrenal insufficiency - The patient's history of hypokalemia and the presence of an adrenal adenoma suggest this diagnosis.
* Pneumonia - The patient's history of cough and fever suggest this diagnosis, but it cannot be confirmed based on the available information.

It is important to note that this assessment is based on the information provided and may not represent the complete clinical picture. Further evaluation and diagnostic testing may be necessary to confirm these diagnoses and rule out other potential conditions."""

        self.evaluator.parse_diagnosis(prediction)

        self.assertEqual(
            self.evaluator.answers["Diagnosis"],
            "Chronic Myeloid Leukemia (CML)",
        )
        self.assertEqual(self.evaluator.scores["Diagnosis Parsing"], 1)

    def test_parse_diagnosis_llama_2_chat_3(self):
        prediction = """Final Diagnosis: Based on the patient's history, physical examination, laboratory results, imaging findings, and the absence of any other explanatory conditions, the most likely diagnosis for this patient is Acute Pancreatitis.

The patient's symptoms of epigastric pain, nausea, vomiting, fever, and abdominal tenderness are consistent with acute pancreatitis. The elevated serum amylase and lipase levels support this diagnosis. Additionally, the CT scan shows a small granuloma in segment VI of the liver, which may indicate chronic pancreatitis. The presence of gallstones and a distended gallbladder also increase the likelihood of acute pancreatitis. The absence of any other explanation for the patient's symptoms and laboratory results further supports this diagnosis."""

        self.evaluator.parse_diagnosis(prediction)

        self.assertEqual(self.evaluator.answers["Diagnosis"], "Acute Pancreatitis")
        self.assertEqual(self.evaluator.scores["Diagnosis Parsing"], 1)

    def test_parse_diagnosis_llama_2_chat_4(self):
        prediction = """Final Diagnosis: Based on the patient's history of persistent right-sided abdominal pain, recent onset of dysuria and urinary frequency, and the finding of a small, non-inflamed appendix on CT scan, it is likely that the patient has acute appendicitis. The presence of a corpus luteum cyst in the left ovary may also be relevant to the patient's overall health and potential treatment plan, but it is not directly related to the likely diagnosis of acute appendicitis. Therefore, the final diagnosis is:

* Acute appendicitis

It is important to note that this diagnosis is based on the information provided and may change depending on the results of further diagnostic tests or the patient's response to treatment."""

        self.evaluator.parse_diagnosis(prediction)

        self.assertEqual(self.evaluator.answers["Diagnosis"], "acute appendicitis")
        self.assertEqual(self.evaluator.scores["Diagnosis Parsing"], 1)

    def test_parse_diagnosis_llama_2_chat_5(self):
        prediction = """Final Diagnosis: Acute Appendicitis. 

The patient's symptoms of sudden-onset right lower quadrant pain, anorexia, nausea, vomiting, and fever, along with physical examination findings of tenderness and guarding in the right lower abdomen, suggest acute appendicitis. This suspicion is supported by the laboratory results showing a slight increase in white blood cell count and neutrophils. The CT scan also shows evidence of appendiceal inflammation and enlarged lymph nodes near the ascending colon. Additionally, the absence of other significant medical conditions in the patient's past medical history and family history supports this diagnosis. Therefore, acute appendicitis is the most likely diagnosis."""

        self.evaluator.parse_diagnosis(prediction)

        self.assertEqual(self.evaluator.answers["Diagnosis"], "Acute Appendicitis")
        self.assertEqual(self.evaluator.scores["Diagnosis Parsing"], 1)

    def test_parse_diagnosis_llama_2_chat_6(self):
        prediction = """Final Diagnosis: Based on the patient's symptoms, laboratory results, and imaging findings, the most likely diagnosis is acute pancreatitis. The patient's abdominal pain, nausea, vomiting, fever, and chills are consistent with this diagnosis. The laboratory results show an elevated white blood cell count, which is also consistent with acute pancreatitis. Additionally, the CT scan shows inflammation in the pancreas, which supports this diagnosis.

The patient's history of chronic celiac artery occlusion and previous episodes of acute pancreatitis also increase the likelihood of this diagnosis. Furthermore, the patient's recent use of vancomycin, which can cause pancreatitis as a side effect, adds to the suspicion of acute pancreatitis.

Therefore, the final diagnosis is acute pancreatitis."""

        self.evaluator.parse_diagnosis(prediction)

        self.assertEqual(self.evaluator.answers["Diagnosis"], "acute pancreatitis")
        self.assertEqual(self.evaluator.scores["Diagnosis Parsing"], 1)

    def test_parse_diagnosis_llama_2_chat_7(self):
        prediction = """Final Diagnosis: Based on the patient's history and current symptoms, the most likely diagnosis is a urinary tract infection (UTI) with a possibility of pyelonephritis. The patient has a history of recurrent UTIs, and the current symptoms of fever, chills, and flank pain are consistent with a UTI. Additionally, the patient has a history of embolization of the donated kidney, which increases the risk of UTI. The patient's recent admission for fever and the absence of any other obvious causes of fever support this diagnosis. It is important to note that the patient's history of multiple admissions for fever of unknown origin and the failure to identify a source for the fever during previous hospitalizations suggest that the patient may have a complex underlying condition that requires further evaluation."""

        self.evaluator.parse_diagnosis(prediction)

        self.assertEqual(
            self.evaluator.answers["Diagnosis"],
            "a urinary tract infection (UTI) with a possibility of pyelonephritis",
        )
        self.assertEqual(self.evaluator.scores["Diagnosis Parsing"], 1)

    def test_parse_diagnosis_llama_2_chat_8(self):
        prediction = """Final Diagnosis: Based on the patient's symptoms, laboratory results, and imaging findings, the most likely diagnosis is acute cholecystitis with a gallstone lodged in the neck of the gallbladder, which is causing intermittent fevers, early satiety, occasional nausea, and constipation. Additionally, the patient has a history of end-stage renal disease secondary to diabetic nephropathy and hypertension, and has undergone a renal transplant. The patient also has a history of malignant melanoma of the left upper chest, but there is no evidence of recurrence.

The patient's urinary tract infection and recent onset of lower abdominal pain are likely related to the acute cholecystitis. The patient's elevated liver enzymes and bilirubin levels are consistent with biliary obstruction caused by the gallstone. The patient's low serum albumin level may be due to malnutrition or nephrotic syndrome. The patient's history of hypertension and diabetes mellitus increases the risk of cardiovascular disease, which should be considered in the patient's management plan.

It is important to note that the patient's history of malignant melanoma and recent onset of lower abdominal pain raise the possibility of an occult primary cancer in the abdomen, which should be ruled out with further imaging studies such as a CT scan of the abdomen and pelvis."""

        self.evaluator.parse_diagnosis(prediction)

        self.assertEqual(
            self.evaluator.answers["Diagnosis"],
            "acute cholecystitis with a gallstone lodged in the neck of the gallbladder",
        )
        self.assertEqual(self.evaluator.scores["Diagnosis Parsing"], 1)

    def test_parse_diagnosis_multi_diag_multi_line_numbered(self):
        prediction = """Final Diagnosis: Based on the patient's symptoms, physical examination, laboratory results, imaging findings, and medical history, the most likely final diagnosis is:

1. Acute appendicitis
2. Prostate cancer with brachytherapy seeds
3. Chronic kidney disease (CKD) stage 3
4. Hypokalemia
5. Hypernatremia"""
        self.evaluator.parse_diagnosis(prediction)

        self.assertEqual(self.evaluator.answers["Diagnosis"], "Acute appendicitis")
        self.assertEqual(self.evaluator.scores["Diagnosis Parsing"], 1)

    def test_parse_diagnosis_multi_diag_multi_line_numbered_first_line(self):
        prediction = """Final Diagnosis: 1. Acute appendicitis
2. Prostate cancer with brachytherapy seeds
3. Chronic kidney disease (CKD) stage 3
4. Hypokalemia
5. Hypernatremia"""
        self.evaluator.parse_diagnosis(prediction)

        self.assertEqual(self.evaluator.answers["Diagnosis"], "Acute appendicitis")
        self.assertEqual(self.evaluator.scores["Diagnosis Parsing"], 1)

    @unittest.skip("Check removed")
    def test_parse_diagnosis_initial_instructions_in_response(self):
        prediction = """
    You are a medical artificial intelligence assistant. You give helpful, detailed and factually correct answers to the doctors questions to help him in his clinical duties. Your goal is to correctly diagnose the patient and provide treatment adivce. You will consider information about a patient and provide a final diagnosis.

    You can only respond with a single complete
    "Thought, Action, Action Input" format
    OR a single "Final Diagnosis, Treatment" format.

    Format 1:

    Thought: (reflect on your progress and decide what to do next)
    Action: (the action name, should be one of [Physical Examination, Laboratory Tests, Imaging])
    Action Input: (the input string to the action)

    OR

    Format 2:

    Final Diagnosis: (the final diagnosis to the original case)
    Treatment: (the treatment for the given diagnosis)

    Human: 
    Diagnose the patient below using the following tools and follow the formats provided above:

        Physical Examination: Perform physical examination of patient and return observations.
        Laboratory Tests: Laboratory Tests. The specific tests must be specified in the 'Action Input' field.
        Imaging: Imaging. Scan region AND modality must be specified in the 'Action Input' field.

    Use the tools provided, using the most specific tool available for each action.
    Provide your response following the Thought, Action, Action Input format or the Final Diagnosis, Treatment format.
    Your final answer should contain all information that lead you to your diagnosis and suggested treatment.

    Patient: Abdominal pain

    AI:
    Thought: I don't know
    """

        with self.assertRaises(ValueError):
            self.evaluator.parse_diagnosis(prediction)

    #############
    # TREATMENT #
    #############

    def test_evaluator_treatment_parsing(self):
        action = AgentAction(
            tool="",
            tool_input={},
            log="",
            custom_parsings=0,
        )
        result = {
            "input": "",
            "output": "Final Diagnosis: Acute Appendicitis\nTreatment: Appendectomy",
            "intermediate_steps": [(action, "")],
        }

        eval = self.evaluator._evaluate_agent_trajectory(
            prediction=result["output"],
            input=result["input"],
            reference=self.base_app_reference,
            agent_trajectory=result["intermediate_steps"],
        )

        self.assertEqual(eval["scores"]["Treatment Parsing"], 0)

        self.assertEqual(eval["answers"]["Treatment"], "Appendectomy")

        self.assertEqual(eval["answers"]["Treatment Required"]["Appendectomy"], False)
        self.assertEqual(eval["answers"]["Treatment Required"]["Antibiotics"], True)
        self.assertEqual(eval["answers"]["Treatment Required"]["Support"], True)

        self.assertEqual(eval["answers"]["Treatment Requested"]["Appendectomy"], True)
        for treatment in eval["answers"]["Treatment Requested"].keys():
            if treatment != "Appendectomy":
                self.assertEqual(
                    eval["answers"]["Treatment Requested"][treatment], False
                )

    def test_evaluator_treatment_parsing_treatment_custom_parsing(self):
        action = AgentAction(
            tool="",
            tool_input={},
            log="",
            custom_parsings=0,
        )
        result = {
            "input": "",
            "output": "Final Diagnosis: Acute Appendicitis\nTreatment Plan: Appendectomy",
            "intermediate_steps": [(action, "")],
        }

        eval = self.evaluator._evaluate_agent_trajectory(
            prediction=result["output"],
            input=result["input"],
            reference=self.base_app_reference,
            agent_trajectory=result["intermediate_steps"],
        )

        self.assertEqual(eval["scores"]["Treatment Parsing"], 1)

        self.assertEqual(eval["answers"]["Treatment"], "Appendectomy")

        self.assertEqual(eval["answers"]["Treatment Required"]["Appendectomy"], False)
        self.assertEqual(eval["answers"]["Treatment Required"]["Antibiotics"], True)
        self.assertEqual(eval["answers"]["Treatment Required"]["Support"], True)

        self.assertEqual(eval["answers"]["Treatment Requested"]["Appendectomy"], True)
        for treatment in eval["answers"]["Treatment Requested"].keys():
            if treatment != "Appendectomy":
                self.assertEqual(
                    eval["answers"]["Treatment Requested"][treatment], False
                )

    ##############
    # ROBUSTNESS #
    ##############

    def test_score_custom_parsing(self):
        steps = []
        tests_ordered = self.evaluator.required_lab_tests["Inflammation"].copy()
        action = AgentAction(
            tool="Laboratory Tests",
            tool_input={"action_input": tests_ordered},
            log="",
            custom_parsings=1,
        )
        observation = ""
        steps.append((action, observation))

        action = AgentAction(
            tool="Physical Examination",
            tool_input={"action_input": None},
            log="",
            custom_parsings=1,
        )
        observation = ""
        steps.append((action, observation))

        action = AgentAction(
            tool="Imaging",
            tool_input={"action_input": {"region": "Abdomen", "modality": "CT"}},
            log="",
            custom_parsings=1,
        )
        observation = ""
        steps.append((action, observation))

        result = {
            "input": "",
            "output": "",
            "intermediate_steps": steps,
        }
        eval = self.evaluator._evaluate_agent_trajectory(
            prediction=result["output"],
            input=result["input"],
            reference=self.base_app_reference,
            agent_trajectory=result["intermediate_steps"],
        )

        self.assertEqual(eval["scores"]["Action Parsing"], 1)

    def test_score_invalid_tools(self):
        steps = []

        action_error = InvalidActionError("Invalid tool", 0)
        action = action_error.invalid_agent_action
        observation = ""
        steps.append((action, observation))

        action = AgentAction(
            tool="Physical Examination",
            tool_input={"action_input": None},
            log="",
            custom_parsings=0,
        )
        observation = ""
        steps.append((action, observation))

        action_error = InvalidActionError("Invalid tool", 0)
        action = action_error.invalid_agent_action
        observation = ""
        steps.append((action, observation))

        result = {
            "input": "",
            "output": "",
            "intermediate_steps": steps,
        }
        eval = self.evaluator._evaluate_agent_trajectory(
            prediction=result["output"],
            input=result["input"],
            reference=self.base_app_reference,
            agent_trajectory=result["intermediate_steps"],
        )

        self.assertEqual(eval["scores"]["Invalid Tools"], 2)

    ##############
    # EFFICIENCY #
    ##############

    def test_score_rounds(self):
        steps = []
        for _ in range(5):
            action = AgentAction(
                tool="Physical Examination",
                tool_input={"action_input": None},
                log="",
                custom_parsings=1,
            )
            observation = ""
            steps.append((action, observation))

        result = {
            "input": "",
            "output": "",
            "intermediate_steps": steps,
        }
        eval = self.evaluator._evaluate_agent_trajectory(
            prediction=result["output"],
            input=result["input"],
            reference=self.base_app_reference,
            agent_trajectory=result["intermediate_steps"],
        )

        self.assertEqual(eval["scores"]["Rounds"], 5)


if __name__ == "__main__":
    unittest.main()
