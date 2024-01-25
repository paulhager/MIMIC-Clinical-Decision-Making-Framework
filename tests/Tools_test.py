import unittest
import pickle

from tools.Actions import (
    get_action_results,
    Actions,
    retrieve_physical_examination,
    retrieve_lab_tests,
    retrieve_imaging,
)
from tools.Tools import RunLaboratoryTests, RunImaging, DoPhysicalExamination
from tests.DummyData import patient_x
from agents.AgentAction import AgentAction


class TestTools(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.action_results = patient_x
        self.lab_test_mapping_path = ""
        with open(self.lab_test_mapping_path, "rb") as f:
            self.lab_test_mapping_df = pickle.load(f)

    def test_DoPhysicalExamination(self):
        tool = DoPhysicalExamination(action_results=self.action_results)
        action = AgentAction(
            tool="Physical Examination",
            tool_input={"action_input": None},
            log="",
            custom_parsings=0,
        )

        output = tool.run(tool_input=action.tool_input)
        expected_output = (
            f"Physical Examination:\n{self.action_results['Physical Examination']}\n"
        )
        self.assertEqual(output, expected_output)

    def test_RunLaboratoryTests(self):
        tool = RunLaboratoryTests(
            action_results=self.action_results,
            lab_test_mapping_df=self.lab_test_mapping_df,
        )
        action = AgentAction(
            tool="Laboratory Tests",
            tool_input={
                "action_input": [51279, 51301]
            },  # Red Blood Cells, White Blood Cells
            log="",
            custom_parsings=0,
        )

        output = tool.run(tool_input=action.tool_input)
        expected_output = (
            "Laboratory Tests:\n"
            f"(Blood) Red Blood Cells: {self.action_results['Laboratory Tests'][51279]}\n"
            f"(Blood) White Blood Cells: {self.action_results['Laboratory Tests'][51301]}\n"
        )
        self.assertEqual(output, expected_output)

    def test_RunImaging(self):
        tool = RunImaging(action_results=self.action_results)
        action_input = {"modality": "CT", "region": "Abdomen"}
        action = AgentAction(
            tool="Imaging",
            tool_input={"action_input": action_input},
            log="",
            custom_parsings=0,
        )

        output = tool.run(tool_input=action.tool_input)
        expected_output = (
            f"Imaging:\nAbdomen CT: {self.action_results['Radiology'][0]['Report']}\n"
        )

        self.assertEqual(output, expected_output)

    def test_get_action_results_physical_examination(self):
        action = Actions.Physical_Examination
        action_input = None
        output = get_action_results(
            action=action,
            action_results=self.action_results,
            action_input=action_input,
        )
        expected_output = (
            f"Physical Examination:\n{self.action_results['Physical Examination']}\n"
        )
        self.assertEqual(output, expected_output)

    def test_get_action_results_lab_tests(self):
        action = Actions.Laboratory_Tests
        action_input = [51279, 51301]
        output = get_action_results(
            action=action,
            action_results=self.action_results,
            action_input=action_input,
            lab_test_mapping_df=self.lab_test_mapping_df,
        )
        expected_output = (
            "Laboratory Tests:\n"
            f"(Blood) Red Blood Cells: {self.action_results['Laboratory Tests'][51279]}\n"
            f"(Blood) White Blood Cells: {self.action_results['Laboratory Tests'][51301]}\n"
        )
        self.assertEqual(output, expected_output)

    def test_get_action_results_imaging(self):
        action = Actions.Imaging
        action_input = {"modality": "CT", "region": "Abdomen"}
        output = get_action_results(
            action=action,
            action_results=self.action_results,
            action_input=action_input,
            already_requested_scans={},
        )
        expected_output = (
            f"Imaging:\nAbdomen CT: {self.action_results['Radiology'][0]['Report']}\n"
        )
        self.assertEqual(output, expected_output)

    def test_retrieve_physical_examination(self):
        result_string = "Physical Examination:\n"

        result_string += retrieve_physical_examination(
            action_results=self.action_results
        )
        expected_output = (
            f"Physical Examination:\n{self.action_results['Physical Examination']}\n"
        )
        self.assertEqual(result_string, expected_output)

    def test_retrieve_lab_results(self):
        result_string = "Laboratory Tests:\n"
        action_input = [
            51279,  # Red Blood Cells
            51755,  # White Blood Cells
            51300,
            51301,
        ]

        result_string += retrieve_lab_tests(
            action_input=action_input,
            action_results=self.action_results,
            lab_test_mapping_df=self.lab_test_mapping_df,
            include_ref_range=False,
            bin_lab_results=False,
        )
        expected_output = (
            "Laboratory Tests:\n"
            f"(Blood) Red Blood Cells: {self.action_results['Laboratory Tests'][51279]}\n"
            f"(Blood) White Blood Cells: {self.action_results['Laboratory Tests'][51301]}\n"
        )
        self.assertEqual(result_string, expected_output)

    def test_retrieve_lab_results_ref_range_included(self):
        result_string = "Laboratory Tests:\n"
        action_input = [
            51279,  # Red Blood Cells
            51755,  # White Blood Cells
            51300,
            51301,
        ]

        result_string += retrieve_lab_tests(
            action_input=action_input,
            action_results=self.action_results,
            lab_test_mapping_df=self.lab_test_mapping_df,
            include_ref_range=True,
            bin_lab_results=False,
        )
        expected_output = (
            "Laboratory Tests:\n"
            f"(Blood) Red Blood Cells: {self.action_results['Laboratory Tests'][51279]} | RR: [4.2 - 5.9]\n"
            f"(Blood) White Blood Cells: {self.action_results['Laboratory Tests'][51301]} | RR: [4.5 - 11.0]\n"
        )
        self.assertEqual(result_string, expected_output)

    def test_retrieve_lab_results_alt_name(self):
        result_string = "Laboratory Tests:\n"
        action_input = [51755, 51300, 51301]  # ["White Blood Cells"]

        action_results = {
            "Laboratory Tests": {
                51300: "7.6 K/uL",  # WBC Count
            }
        }

        result_string += retrieve_lab_tests(
            action_input=action_input,
            action_results=action_results,
            lab_test_mapping_df=self.lab_test_mapping_df,
            include_ref_range=False,
            bin_lab_results=False,
        )

        expected_output = (
            "Laboratory Tests:\n"
            f"(Blood) WBC Count: {action_results['Laboratory Tests'][51300]}\n"
        )
        self.assertEqual(result_string, expected_output)

    def test_retrieve_lab_results_not_available(self):
        result_string = "Laboratory Tests:\n"
        action_input = [
            51279,  # Red Blood Cells
            51755,  # White Blood Cells
            51300,
            51301,
            "Thyroxine",
        ]

        result_string += retrieve_lab_tests(
            action_input=action_input,
            action_results=self.action_results,
            lab_test_mapping_df=self.lab_test_mapping_df,
            include_ref_range=False,
            bin_lab_results=False,
        )
        expected_output = (
            "Laboratory Tests:\n"
            f"(Blood) Red Blood Cells: {self.action_results['Laboratory Tests'][51279]}\n"
            f"(Blood) White Blood Cells: {self.action_results['Laboratory Tests'][51301]}\n"
            "Thyroxine: N/A\n"
        )
        self.assertEqual(result_string, expected_output)

    def test_retrieve_imaging(self):
        result_string = "Imaging:\n"
        action_input = {"modality": "CT", "region": "Abdomen"}

        result_string += retrieve_imaging(
            action_input=action_input,
            action_results=self.action_results,
            already_requested_scans={},
        )
        expected_output = (
            f"Imaging:\nAbdomen CT: {self.action_results['Radiology'][0]['Report']}\n"
        )
        self.assertEqual(result_string, expected_output)

    def test_retrieve_imaging_unique_to_broad(self):
        result_string = "Imaging:\n"
        action_input = {"modality": "MRI", "region": "Abdomen"}

        self.action_results["Radiology"].append(
            {"Modality": "MRCP", "Region": "Abdomen", "Report": "Great MRCP scan."}
        )

        result_string += retrieve_imaging(
            action_input=action_input,
            action_results=self.action_results,
            already_requested_scans={},
        )
        expected_output = "Imaging:\nAbdomen MRI: Great MRCP scan.\n"
        self.assertEqual(result_string, expected_output)

    def test_retrieve_imaging_not_available(self):
        result_string = "Imaging:\n"
        action_input = {"modality": "CT", "region": "Head"}

        result_string += retrieve_imaging(
            action_input=action_input,
            action_results=self.action_results,
            already_requested_scans={},
        )
        expected_output = (
            "Imaging:\nHead CT: Not available. Try a different imaging modality.\n"
        )
        self.assertEqual(result_string, expected_output)

    def test_retrieve_imaging_repeat(self):
        result_string = "Imaging:\n"
        action_input = {"modality": "CT", "region": "Abdomen"}

        repeat_results = self.action_results.copy()
        repeat_results["Radiology"].append(
            {"Modality": "CT", "Region": "Abdomen", "Report": "Repeat scan."}
        )

        result_string += retrieve_imaging(
            action_input=action_input,
            action_results=repeat_results,
            already_requested_scans={"Abdomen CT": 1},
        )
        expected_output = "Imaging:\nAbdomen CT: Repeat scan.\n"
        self.assertEqual(result_string, expected_output)

    def test_retrieve_imaging_no_repeat(self):
        result_string = "Imaging:\n"
        action_input = {"modality": "CT", "region": "Abdomen"}

        result_string += retrieve_imaging(
            action_input=action_input,
            action_results=self.action_results,
            already_requested_scans={"Abdomen CT": 1},
        )
        expected_output = "Imaging:\nAbdomen CT: Cannot repeat this scan anymore. Try a different imaging modality.\n"
        self.assertEqual(result_string, expected_output)

    def test_retrieve_imaging_no_repeat_ask_twice(self):
        tool = RunImaging(action_results=self.action_results)
        action_input = {"modality": "CT", "region": "Abdomen"}
        action = AgentAction(
            tool="Imaging",
            tool_input={"action_input": action_input},
            log="",
            custom_parsings=0,
        )

        output = tool.run(tool_input=action.tool_input)
        expected_output = (
            f"Imaging:\nAbdomen CT: {self.action_results['Radiology'][0]['Report']}\n"
        )

        self.assertEqual(output, expected_output)

        output = tool.run(tool_input=action.tool_input)
        expected_output = "Imaging:\nAbdomen CT: Cannot repeat this scan anymore. Try a different imaging modality.\n"

        self.assertEqual(output, expected_output)


if __name__ == "__main__":
    unittest.main()
