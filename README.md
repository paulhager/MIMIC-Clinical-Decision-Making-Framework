# MIMIC Clinical Decision Making Framework

For a video overview of the paper, checkout this talk I held for the BZKF: https://www.youtube.com/watch?v=sCDaUC16mHA

## News

**🔥 New Addition: Llama 3.3 has been added to the leaderboard! 🔥**

**🔥 New Addition: Llama 3.1 has been added to the leaderboard! 🔥**

**🔥 New Addition: OpenBio has been added to the leaderboard! 🔥**

**🔥 New Addition: Llama 3 has been added to the leaderboard! 🔥**

## Overview

This repository contains the code for running the clinical decision making task using the MIMIC CDM dataset.

The code to create the dataset is found at: https://github.com/paulhager/MIMIC-Clinical-Decision-Making-Dataset

The dataset is based on the MIMIC-IV database. Access can be requested here: https://physionet.org/content/mimiciv/2.2/

A pre-processed version of the dataset is found here: https://physionet.org/content/mimic-iv-ext-cdm/

Visit https://huggingface.co/spaces/MIMIC-CDM/leaderboard to check out the current leaderboard. I will update this as new models are released. If you would like a model to be tested and put on the board, please write me an email at paul (dot) hager (at) tum (dot) de.

## MIMIC CDM

This code simulates a realistic clinical environment where an LLM is provided with the history of present illness of a patient and then tasked to gather information to come to a final diagnosis and treatment plan.

To run the clinical decision making task, execute ```python run.py```. The arguments for this file are specified through config files managed by the hydra library and found under [configs](./configs/). The most important arguments are:
- pathology: Specify one of appendicitis, cholecystitis, diverticulitis, pancreatitis
- model: Specify which model to use. The model file also contains the different role tags
- summarize: Automatically summarize the progress if we begin to reach the token limit

These additional arguments change the way information is presented but did not help performance in my experience and so were not included in the paper:
- include_ref_range: Include the reference ranges for lab results, as provided in the MIMIC database
- bin_lab_results: Replace exact lab result values with the word "low", "normal", or "high", using the reference ranges
- provide_diagnostic_criteria: Adds an extra tool where the model can consult diagnostic criteria if desired
- diag_crit_writer_openai_api_key: OpenAI key to ask for new diagnostic criteria if they are missing from the datafile
- include_tool_use_examples: Provides examples of how to use the tools

## MIMIC CDM Full Information

For the MIMIC-CDM-Full Information task, executed through ```python run_full_info.py```, all relevant information required for a diagnosis is provided upfront to the model and only a diagnosis is asked for. This allows us to also control what information we provide the model and explore many aspects of model performance such as robustness. The relevant arguments for this task are those from above and additionally:
- prompt_template: Determines the system instruction or prompt used to ask for an answer. Possible values are specified in run_full_info.py
- order: The order in which information is provided
- abbreviated: Provide the original, abbreviated text
- fewshot: Provides hand-crafted fewshot cases and diagnosis examples
- save_probabilities: Saves the probabilities of the generation for downstream analysis
- only_abnormal_labs: Provide only those lab results that are abnormal.
- bin_lab_results_abnormal: If only abnormal labs are provided, also bin them

## Other

Housekeeping arguments are:
- seed: The seed used for greedy decoding
- local_logging: If logs should be saved locally
- run_descr: An extra name to give to the run
- first_patient: Start executing at a specific patient
- patient_list_path: Run on only a select group of patients (given as a list of hadm_ids)

## Environment

To setup the environment, create a new virtual environment of your choosing with python=3.10, export your CUDA_HOME path to whatever version CUDA you have (does not have to be 11.7.1 like in the example) and then install the libraries from requirements.txt:

```
export CUDA_HOME=.../cuda/cuda_11.7.1
pip install --no-deps -r requirements.txt
```


# Citation

If you found this code and dataset useful, please cite our paper and dataset with:

Hager, P., Jungmann, F., Holland, R. et al. Evaluation and mitigation of the limitations of large language models in clinical decision-making. Nat Med (2024). https://doi.org/10.1038/s41591-024-03097-1
```
@article{hager_evaluation_2024,
	title = {Evaluation and mitigation of the limitations of large language models in clinical decision-making},
	issn = {1546-170X},
	url = {https://doi.org/10.1038/s41591-024-03097-1},
	doi = {10.1038/s41591-024-03097-1},,
	journaltitle = {Nature Medicine},
	shortjournal = {Nature Medicine},
	author = {Hager, Paul and Jungmann, Friederike and Holland, Robbie and Bhagat, Kunal and Hubrecht, Inga and Knauer, Manuel and Vielhauer, Jakob and Makowski, Marcus and Braren, Rickmer and Kaissis, Georgios and Rueckert, Daniel},
	date = {2024-07-04},
}
```

Hager, P., Jungmann, F., & Rueckert, D. (2024). MIMIC-IV-Ext Clinical Decision Making: A MIMIC-IV Derived Dataset for Evaluation of Large Language Models on the Task of Clinical Decision Making for Abdominal Pathologies (version 1.0). PhysioNet. https://doi.org/10.13026/2pfq-5b68.
```
@misc{hager_mimic-iv-ext_nodate,
	title = {{MIMIC}-{IV}-Ext Clinical Decision Making: A {MIMIC}-{IV} Derived Dataset for Evaluation of Large Language Models on the Task of Clinical Decision Making for Abdominal Pathologies},
	url = {https://physionet.org/content/mimic-iv-ext-cdm/1.0/},
	shorttitle = {{MIMIC}-{IV}-Ext Clinical Decision Making},
	publisher = {{PhysioNet}},
	author = {Hager, Paul and Jungmann, Friederike and Rueckert, Daniel},
	urldate = {2024-07-04},
	doi = {10.13026/2PFQ-5B68},
	note = {Version Number: 1.0
Type: dataset},
}
```
