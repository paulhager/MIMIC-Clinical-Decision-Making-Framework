import copy
import pickle
import unittest

from utils.nlp import (
    convert_labs_to_itemid,
    remove_stop_words,
    extract_sections,
    extract_primary_diagnosis,
    create_lab_test_string,
)
from tests.DummyData import patient_x


class TestNLPUtils(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
        with open("", "rb") as f:
            self.lab_test_mapping_df = pickle.load(f)

        self.cbc_tests = self.lab_test_mapping_df.loc[
            self.lab_test_mapping_df["label"] == "Complete Blood Count (CBC)",
            "corresponding_ids",
        ].squeeze()
        self.rfp_tests = self.lab_test_mapping_df.loc[
            self.lab_test_mapping_df["label"] == "Renal Function Panel (RFP)",
            "corresponding_ids",
        ].squeeze()

        self.lfp_tests = self.lab_test_mapping_df.loc[
            self.lab_test_mapping_df["label"] == "Liver Function Panel (LFP)",
            "corresponding_ids",
        ].squeeze()

        self.tags = {
            "system_tag_start": "<|im_start|>system",
            "system_tag_end": "<|im_end|>",
            "user_tag_start": "<|im_start|>user",
            "user_tag_end": "<|im_end|>",
            "ai_tag_start": "<|im_start|>assistant",
            "ai_tag_end": "<|im_end|>",
        }

    ###########################
    # remove_stop_words tests #
    ###########################

    def test_remove_stop_words(self):
        sentence = "This is a sentence with stop words."
        output = remove_stop_words(sentence)

        expected_output = "sentence stop words."

        self.assertEqual(output, expected_output)

    def test_remove_stop_words_with_punctuation(self):
        sentence = "This is a sentence with stop words, and punctuation."
        output = remove_stop_words(sentence)

        expected_output = "sentence stop words, punctuation."

        self.assertEqual(output, expected_output)

    def test_remove_stop_words_A_in_test(self):
        sentence = (
            "This is a sentence with stop words, and punctuation. and the test Glyco A"
        )
        output = remove_stop_words(sentence)

        expected_output = "sentence stop words, punctuation. test Glyco A"

        self.assertEqual(output, expected_output)

    def test_remove_stop_words_A_prefix_in_test(self):
        sentence = "A CBC. This is a sentence with stop words, and punctuation. and the test Glyco A"
        output = remove_stop_words(sentence)

        expected_output = "CBC. sentence stop words, punctuation. test Glyco A"

        self.assertEqual(output, expected_output)

    ###################################
    # extract_primary_diagnosis tests #
    ###################################

    def test_extract_primary_diagnosis_vanilla(self):
        text = """diagnosis:
gallstone pancreatitis
choledocholithiasis
possible cholangitis"""
        output = extract_primary_diagnosis(text)
        self.assertEqual(output, "gallstone pancreatitis")

    def test_extract_primary_diagnosis_vanilla_2(self):
        text = """diagnosis:
primary diagnosis
acute uncomplicated diverticulitis
acute kidney injury 

secondary diagnosis
heart failure with reduced ejection fraction"""

        output = extract_primary_diagnosis(text)
        self.assertEqual(output, "acute uncomplicated diverticulitis")

    def test_extract_primary_diagnosis_primary_diagnosis(self):
        text = """diagnosis:
primary diagnosis
=================
- epigastric abdominal pain
- pancreatitis, mild

secondary diagnoses
====================
- s/p renal transplant, on immunosuppression"""
        output = extract_primary_diagnosis(text)
        self.assertEqual(output, "epigastric abdominal pain")

    def test_extract_primary_diagnosis_comma_seperated(self):
        text = """diagnosis:
symptomatic cholelithiasis, possible gallstone pancreatitis"""

        output = extract_primary_diagnosis(text)
        self.assertEqual(output, "symptomatic cholelithiasis")

    def test_extract_primary_diagnoses(self):
        text = """diagnoses:
symptomatic cholelithiasis
possible gallstone pancreatitis"""

        output = extract_primary_diagnosis(text)
        self.assertEqual(output, "symptomatic cholelithiasis")

    def test_extract_primary_diagnosis_vs(self):
        text = """diagnosis:
- acute pancreatitis due to medications vs benign ampullary 
stenosis
- acute metabolic encephalopathy/delirium, multifactorial"""
        output = extract_primary_diagnosis(text)
        self.assertEqual(output, "acute pancreatitis due to medications")

    def test_extract_primary_diagnosis_vs_period(self):
        text = """diagnosis:
- acute pancreatitis due to medications vs. benign ampullary 
stenosis
- acute metabolic encephalopathy/delirium, multifactorial"""
        output = extract_primary_diagnosis(text)
        self.assertEqual(output, "acute pancreatitis due to medications")

    def test_extract_primary_diagnosis_and(self):
        text = """diagnosis:
severe sepsis due to cholangitis and gallstone pancreatitis with 
klebsiella bacteremia. cholecystitis

heart failure/pulmonary edema """
        output = extract_primary_diagnosis(text)
        self.assertEqual(output, "severe sepsis due to cholangitis")

    def test_extract_primary_diagnosis_or(self):
        text = """diagnosis:
small bowel obstruction secondary to small bowel tumor or perforated diverticulitis of the small bowel."""

        output = extract_primary_diagnosis(text)
        self.assertEqual(
            output, "small bowel obstruction secondary to small bowel tumor"
        )

    def test_extract_primary_diagnosis_and_substring(self):
        text = """diagnosis:
        bandemia and leukocytosis, possible sepsis"""

        output = extract_primary_diagnosis(text)
        self.assertEqual(output, "bandemia")

    def test_extract_primary_diagnosis_same_line(self):
        text = """diagnosis: pancreatitis, cholestasis

secondary diagnoses: portal vein thrombosis, gerd, depression"""

        output = extract_primary_diagnosis(text)
        self.assertEqual(output, "pancreatitis")

    def test_extract_primary_diagnosis_numbered(self):
        text = """diagnosis:
1. perforated sigmoid colonic diverticulitis 
2. atrial flutter"""
        output = extract_primary_diagnosis(text)

        self.assertEqual(output, "perforated sigmoid colonic diverticulitis")

    #########################################
    # convert_labs_to_itemid tests #
    #########################################

    def test_exact_match(self):
        tests = ["Amylase"]
        output = convert_labs_to_itemid(tests, self.lab_test_mapping_df)

        expected_output = [50867]

        self.assertEqual(output, expected_output)

    def test_spelling_mistake(self):
        tests = ["Amylasee"]
        output = convert_labs_to_itemid(tests, self.lab_test_mapping_df)

        expected_output = [50867]

        self.assertEqual(output, expected_output)

    def test_different_name_substring(self):
        tests = ["Amylase Dehydrogenase"]
        output = convert_labs_to_itemid(tests, self.lab_test_mapping_df)

        expected_output = ["Amylase Dehydrogenase"]

        # i.e. we do not get amylase
        self.assertNotIn(
            "Amylase Dehydrogenase", list(self.lab_test_mapping_df["label"])
        )
        self.assertEqual(output, expected_output)

    def test_full_exact_expansion(self):
        tests = ["Complete Blood Count (CBC)"]
        output = convert_labs_to_itemid(tests, self.lab_test_mapping_df)

        expected_output = self.cbc_tests

        self.assertEqual(output, expected_output)

    def test_multi_hit_extension(self):
        tests = ["CBC", "Amylase"]
        output = convert_labs_to_itemid(tests, self.lab_test_mapping_df)

        expected_output = self.cbc_tests + [50867]

        self.assertEqual(output, expected_output)

    def test_long_exact_expansion(self):
        tests = ["Complete Blood Count"]
        output = convert_labs_to_itemid(tests, self.lab_test_mapping_df)

        expected_output = self.cbc_tests

        self.assertEqual(output, expected_output)

    def test_short_exact_expansion(self):
        tests = ["CBC"]
        output = convert_labs_to_itemid(tests, self.lab_test_mapping_df)

        expected_output = self.cbc_tests

        self.assertEqual(output, expected_output)

    def test_wrong_long_correct_short_expansion(self):
        tests = ["All Blood Tests (CBC)"]
        output = convert_labs_to_itemid(tests, self.lab_test_mapping_df)

        expected_output = self.cbc_tests

        self.assertEqual(output, expected_output)

    def test_wrong_short_correct_long_expansion(self):
        tests = ["Complete Blood Count (CompBldCnt)"]
        output = convert_labs_to_itemid(tests, self.lab_test_mapping_df)

        expected_output = self.cbc_tests

        self.assertEqual(output, expected_output)

    def test_medium_blood(self):
        tests = ["blood glucose"]
        output = convert_labs_to_itemid(tests, self.lab_test_mapping_df)

        expected_output = [50809, 50931, 52569]

        self.assertEqual(output, expected_output)

    def test_medium_serum(self):
        tests = ["serum glucose"]
        output = convert_labs_to_itemid(tests, self.lab_test_mapping_df)

        expected_output = [50809, 50931, 52569]

        self.assertEqual(output, expected_output)

    def test_medium_in_paranthesis(self):
        tests = ["Amylase (Urine)"]
        output = convert_labs_to_itemid(tests, self.lab_test_mapping_df)

        expected_output = [51072]

        self.assertEqual(output, expected_output)

    def test_medium_in_paranthesis_2(self):
        tests = ["Glucose (Blood)"]
        output = convert_labs_to_itemid(tests, self.lab_test_mapping_df)

        expected_output = [50809, 50931, 52569]

        self.assertEqual(output, expected_output)

    def test_alternate_spelling_expansion_1(self):
        tests = ["Liver Function Tests (LFTs)"]
        output = convert_labs_to_itemid(tests, self.lab_test_mapping_df)

        expected_output = self.lfp_tests

        self.assertEqual(output, expected_output)

    def test_alternate_spelling_expansion_2(self):
        tests = ["Renal Function Tests"]
        output = convert_labs_to_itemid(tests, self.lab_test_mapping_df)

        expected_output = self.rfp_tests

        self.assertEqual(output, expected_output)

    def test_wrong_name(self):
        tests = ["CBD"]
        output = convert_labs_to_itemid(tests, self.lab_test_mapping_df)

        expected_output = tests

        self.assertNotIn("CBD", list(self.lab_test_mapping_df["label"]))
        self.assertEqual(output, expected_output)

    ##########################
    # extract_sections tests #
    ##########################

    def test_extract_sections(self):
        text = "<|im_start|>system\nYou are a medical artificial intelligence assistant. Your goal is to effectively, efficiently and accurately reduce text without inventing information. You want to return verbatim observations that are abnormal and of interest to a possible diagnosis of the patient. Normal observations can be combined. Do not invent information. Use medical abbreviations when possible to save characters. Put the most important information first.\n<|im_end|>\n<|im_start|>user\nPlease summarize the following result:\nPhysical Examination:\n Gen: PAtient with NAD, AAOx3 CVA: S1 and S2, tachy, SR Lungs: CTA ___ Abdomen: soft, nontender, nondistended, +BS, no rebound Extr: no c/c/e    \n\n<|im_end|>\n<|im_start|>assistant\nSummary:very important summary\n<|im_end|>\n<|im_start|>user\nPlease choose a valid tool or provide a Final Diagnosis and Treatment.\n<|im_end|>\n<|im_start|>assistant\nThoughts:"

        output = extract_sections(text, self.tags)

        expected_output = [
            {
                "role": "system",
                "content": "You are a medical artificial intelligence assistant. Your goal is to effectively, efficiently and accurately reduce text without inventing information. You want to return verbatim observations that are abnormal and of interest to a possible diagnosis of the patient. Normal observations can be combined. Do not invent information. Use medical abbreviations when possible to save characters. Put the most important information first.",
            },
            {
                "role": "user",
                "content": "Please summarize the following result:\nPhysical Examination:\n Gen: PAtient with NAD, AAOx3 CVA: S1 and S2, tachy, SR Lungs: CTA ___ Abdomen: soft, nontender, nondistended, +BS, no rebound Extr: no c/c/e",
            },
            {
                "role": "assistant",
                "content": "Summary:very important summary",
            },
            {
                "role": "user",
                "content": "Please choose a valid tool or provide a Final Diagnosis and Treatment.",
            },
            {
                "role": "assistant",
                "content": "Thoughts:",
            },
        ]

        self.assertEqual(output, expected_output)

    ##########################
    # create_lab_test_string #
    ##########################

    def test_create_lab_test_string(self):
        test = 51279
        output = create_lab_test_string(
            test, self.lab_test_mapping_df, patient_x, include_ref_range=False
        )
        expected_output = "(Blood) Red Blood Cells: 4.74 m/uL\n"

        self.assertEqual(output, expected_output)

    def test_create_lab_test_string_microbio(self):
        test = 90201
        output = create_lab_test_string(
            test, self.lab_test_mapping_df, patient_x, include_ref_range=False
        )
        expected_output = "(Microbiology) Blood Culture, Routine: No growth.\n"

        self.assertEqual(output, expected_output)

    def test_create_lab_test_string_rr(self):
        test = 51279
        output = create_lab_test_string(
            test, self.lab_test_mapping_df, patient_x, include_ref_range=True
        )
        expected_output = "(Blood) Red Blood Cells: 4.74 m/uL | RR: [4.2 - 5.9]\n"

        self.assertEqual(output, expected_output)

    def test_create_lab_test_string_rr_na(self):
        test = 50861
        output = create_lab_test_string(
            test, self.lab_test_mapping_df, patient_x, include_ref_range=True
        )
        expected_output = "(Blood) Alanine Aminotransferase (ALT): 18.0 IU/L\n"

        self.assertEqual(output, expected_output)

    def test_create_lab_test_string_binning_normal(self):
        test = 51279
        output = create_lab_test_string(
            test,
            self.lab_test_mapping_df,
            patient_x,
            include_ref_range=False,
            bin_lab_results=True,
        )
        expected_output = "(Blood) Red Blood Cells: Normal\n"

        self.assertEqual(output, expected_output)

    def test_create_lab_test_string_binning_low(self):
        test = 51279
        patient = copy.deepcopy(patient_x)
        patient["Laboratory Tests"][test] = "3.0 m/uL"
        output = create_lab_test_string(
            test,
            self.lab_test_mapping_df,
            patient,
            include_ref_range=False,
            bin_lab_results=True,
        )
        expected_output = "(Blood) Red Blood Cells: Low\n"

        self.assertEqual(output, expected_output)

    def test_create_lab_test_string_binning_high(self):
        test = 51279
        patient = copy.deepcopy(patient_x)
        patient["Laboratory Tests"][test] = "7.5 m/uL"
        output = create_lab_test_string(
            test,
            self.lab_test_mapping_df,
            patient,
            include_ref_range=False,
            bin_lab_results=True,
        )
        expected_output = "(Blood) Red Blood Cells: High\n"

        self.assertEqual(output, expected_output)


if __name__ == "__main__":
    unittest.main()
