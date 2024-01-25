from typing import Union, Dict, Type, List
from pydantic import BaseModel, Field

from langchain.tools import BaseTool
import pandas as pd

from tools.Actions import get_action_results, Actions


class LaboratoryTests_Input(BaseModel):
    action_input: List = Field(description="The names of the lab tests to run.")


class Imaging_Input(BaseModel):
    action_input: Dict = Field(
        description="The names of the imaging modality and region to be scanned."
    )


class RunLaboratoryTests(BaseTool):
    name: str = "Laboratory Tests"
    description: str = "Laboratory Tests. The specific tests must be specified in the 'Action Input' field."
    args_schema: Type[BaseModel] = LaboratoryTests_Input
    action_results: Dict = {}
    lab_test_mapping_df: pd.DataFrame = None
    include_ref_range: bool = False
    bin_lab_results: bool = False

    def _run(self, action_input: List[Union[int, str]]) -> str:
        return get_action_results(
            action=Actions.Laboratory_Tests,
            action_results=self.action_results,
            action_input=action_input,
            lab_test_mapping_df=self.lab_test_mapping_df,
            include_ref_range=self.include_ref_range,
            bin_lab_results=self.bin_lab_results,
        )

    async def _arun(self, action_input: List[Union[int, str]]) -> str:
        return get_action_results(
            action=Actions.Laboratory_Tests,
            action_results=self.action_results,
            action_input=action_input,
            lab_test_mapping_df=self.lab_test_mapping_df,
            include_ref_range=self.include_ref_range,
            bin_lab_results=self.bin_lab_results,
        )


class RunImaging(BaseTool):
    name: str = "Imaging"
    description: str = "Imaging. Scan region AND modality must be specified in the 'Action Input' field."
    args_schema: Type[BaseModel] = Imaging_Input
    action_results: Dict = {}
    already_requested_scans: Dict = {}

    def _run(self, action_input: Dict) -> str:
        return get_action_results(
            action=Actions.Imaging,
            action_results=self.action_results,
            action_input=action_input,
            already_requested_scans=self.already_requested_scans,
        )

    async def _arun(self, action_input: Dict) -> str:
        return get_action_results(
            action=Actions.Imaging,
            action_results=self.action_results,
            action_input=action_input,
            already_requested_scans=self.already_requested_scans,
        )


class DoPhysicalExamination(BaseTool):
    name: str = "Physical Examination"
    description: str = (
        "Perform physical examination of patient and return observations."
    )
    action_results: Dict = {}

    def _run(self, action_input: str) -> str:
        return get_action_results(
            action=Actions.Physical_Examination, action_results=self.action_results
        )

    async def _arun(self, action_input: str) -> str:
        return get_action_results(
            action=Actions.Physical_Examination, action_results=self.action_results
        )


class ReadDiagnosticCriteria(BaseTool):
    name: str = "Diagnostic Criteria"
    description: str = "Read diagnostic criteria for a given pathology."

    def _run(self, action_input: str) -> str:
        return get_action_results(
            action=Actions.Diagnostic_Criteria,
            action_input=action_input,
        )

    async def _arun(self, action_input: str) -> str:
        return get_action_results(
            action=Actions.Diagnostic_Criteria,
            action_input=action_input,
        )
