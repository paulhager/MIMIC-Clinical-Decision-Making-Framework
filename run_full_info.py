import os
import re
from os.path import join
import json
import random
from datetime import datetime
import time
import pickle
import fcntl

import numpy as np
import hydra
from omegaconf import DictConfig
from loguru import logger
import langchain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from thefuzz import process

from utils.nlp import calculate_num_tokens, truncate_text, create_lab_test_string
from dataset.utils import load_hadm_from_file
from utils.logging import append_to_pickle_file
from evaluators.appendicitis_evaluator import AppendicitisEvaluator
from evaluators.cholecystitis_evaluator import CholecystitisEvaluator
from evaluators.diverticulitis_evaluator import DiverticulitisEvaluator
from evaluators.pancreatitis_evaluator import PancreatitisEvaluator
from models.models import CustomLLM
from agents.prompts import (
    FULL_INFO_TEMPLATE,
    FULL_INFO_TEMPLATE_SECTION,
    FULL_INFO_TEMPLATE_NO_SYSTEM,
    FULL_INFO_TEMPLATE_NO_MEDICAL,
    FULL_INFO_TEMPLATE_SERIOUS,
    FULL_INFO_TEMPLATE_MINIMAL_SYSTEM,
    FULL_INFO_TEMPLATE_NO_USER,
    FULL_INFO_TEMPLATE_NO_SYSTEM_NO_USER,
    FULL_INFO_TEMPLATE_NO_PROMPT,
    FULL_INFO_TEMPLATE_NOFINAL,
    FULL_INFO_TEMPLATE_MAINDIAGNOSIS,
    FULL_INFO_TEMPLATE_PRIMARYDIAGNOSIS,
    FULL_INFO_TEMPLATE_ACUTE,
    FULL_INFO_TEMPLATE_COT,
    FULL_INFO_TEMPLATE_COT_FINAL_DIAGNOSIS,
    FULL_INFO_TEMPLATE_TOP3,
    FI_FEWSHOT_TEMPLATE_COPD,
    FI_FEWSHOT_TEMPLATE_PNEUMONIA,
    FI_FEWSHOT_TEMPLATE_COPD_RR,
    FI_FEWSHOT_TEMPLATE_PNEUMONIA_RR,
    SUMMARIZE_OBSERVATION_TEMPLATE,
    DIAGNOSTIC_CRITERIA_APPENDICITIS,
    DIAGNOSTIC_CRITERIA_CHOLECYSTITIS,
    DIAGNOSTIC_CRITERIA_DIVERTICULITIS,
    DIAGNOSTIC_CRITERIA_PANCREATITIS,
    CONFIRM_DIAG_TEMPLATE,
    WRITE_DIAG_CRITERIA_TEMPLATE,
)

gpt_tags = {
    "system_tag_start": "<|im_start|>system",
    "user_tag_start": "<|im_start|>user",
    "ai_tag_start": "<|im_start|>assistant",
    "system_tag_end": "<|im_end|>",
    "user_tag_end": "<|im_end|>",
    "ai_tag_end": "<|im_end|>",
}

STOP_WORDS = []


def load_evaluator(pathology):
    # Load desired evaluator
    if pathology == "appendicitis":
        evaluator = AppendicitisEvaluator()
    elif pathology == "cholecystitis":
        evaluator = CholecystitisEvaluator()
    elif pathology == "diverticulitis":
        evaluator = DiverticulitisEvaluator()
    elif pathology == "pancreatitis":
        evaluator = PancreatitisEvaluator()
    else:
        raise NotImplementedError
    return evaluator


@hydra.main(config_path="./configs", config_name="config", version_base=None)
def run(args: DictConfig):
    if args.self_consistency:
        args.save_probabilities = True
    else:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # Set stop words
    global STOP_WORDS
    STOP_WORDS = args.stop_words

    tags = {
        "system_tag_start": args.system_tag_start,
        "user_tag_start": args.user_tag_start,
        "ai_tag_start": args.ai_tag_start,
        "system_tag_end": args.system_tag_end,
        "user_tag_end": args.user_tag_end,
        "ai_tag_end": args.ai_tag_end,
    }

    # Load desired model
    llm = CustomLLM(
        model_name=args.model_name,
        openai_api_key=args.openai_api_key,
        tags=tags,
        max_context_length=args.max_context_length,
        exllama=args.exllama,
        seed=args.seed,
        self_consistency=args.self_consistency,
    )
    llm.load_model(args.base_models)

    if args.confirm_diagnosis:
        diag_crit_writer = CustomLLM(
            model_name="gpt-3.5-turbo",
            openai_api_key=args.diag_crit_writer_openai_api_key,
            tags=gpt_tags,
            max_context_length=4096,
            exllama=False,
            seed=2023,
        )
        diag_crit_writer.load_model(args.base_models)

    # Interpret desired prompt
    if args.prompt_template == "NOSYSTEM":
        prompt_template = FULL_INFO_TEMPLATE_NO_SYSTEM
    elif args.prompt_template == "NOUSER":
        prompt_template = FULL_INFO_TEMPLATE_NO_USER
    elif args.prompt_template == "NOSYSTEMNOUSER":
        prompt_template = FULL_INFO_TEMPLATE_NO_SYSTEM_NO_USER
    elif args.prompt_template == "NOMEDICAL":
        prompt_template = FULL_INFO_TEMPLATE_NO_MEDICAL
    elif args.prompt_template == "SERIOUS":
        prompt_template = FULL_INFO_TEMPLATE_SERIOUS
    elif args.prompt_template == "MINIMALSYSTEM":
        prompt_template = FULL_INFO_TEMPLATE_MINIMAL_SYSTEM
    elif args.prompt_template == "NOPROMPT":
        prompt_template = FULL_INFO_TEMPLATE_NO_PROMPT
    elif args.prompt_template == "NOFINAL":
        prompt_template = FULL_INFO_TEMPLATE_NOFINAL
    elif args.prompt_template == "MAINDIAGNOSIS":
        prompt_template = FULL_INFO_TEMPLATE_MAINDIAGNOSIS
    elif args.prompt_template == "PRIMARYDIAGNOSIS":
        prompt_template = FULL_INFO_TEMPLATE_PRIMARYDIAGNOSIS
    elif args.prompt_template == "ACUTE":
        prompt_template = FULL_INFO_TEMPLATE_ACUTE
    elif args.prompt_template == "SECTION":
        prompt_template = FULL_INFO_TEMPLATE_SECTION
    elif args.prompt_template == "TOP3":
        prompt_template = FULL_INFO_TEMPLATE_TOP3
        args.save_probabilities = True
    elif args.prompt_template == "COT":
        prompt_template = FULL_INFO_TEMPLATE_COT
        final_diagnosis_prompt = PromptTemplate(
            template=FULL_INFO_TEMPLATE_COT_FINAL_DIAGNOSIS,
            input_variables=["cot"],
            partial_variables={
                "system_tag_start": tags["system_tag_start"],
                "system_tag_end": tags["system_tag_end"],
                "user_tag_start": tags["user_tag_start"],
                "user_tag_end": tags["user_tag_end"],
                "ai_tag_start": tags["ai_tag_start"],
            },
        )
        final_diag_chain = LLMChain(llm=llm, prompt=final_diagnosis_prompt)

    elif args.prompt_template == "VANILLA":
        prompt_template = FULL_INFO_TEMPLATE
    else:
        raise NotImplementedError

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["input", "fewshot_examples", "diagnostic_criteria"],
        partial_variables={
            "system_tag_start": tags["system_tag_start"],
            "system_tag_end": tags["system_tag_end"],
            "user_tag_start": tags["user_tag_start"],
            "user_tag_end": tags["user_tag_end"],
            "ai_tag_start": tags["ai_tag_start"],
        },
    )
    langchain.debug = True

    chain = LLMChain(llm=llm, prompt=prompt)

    date_time = datetime.fromtimestamp(time.time())
    str_date = date_time.strftime("%d-%m-%Y_%H:%M:%S")
    args.model_name = args.model_name.replace("/", "_")
    run_name = f"{args.pathology}_{args.model_name}_{str_date}_FULL_INFO"

    # Create run_name string
    if args.order:
        run_name += f"_{args.order.upper()}"
    else:
        run_name += "_H"
    if args.diagnostic_criteria:
        run_name += f"_{args.diagnostic_criteria.upper()}"
    else:
        run_name += "_N"
    if args.fewshot:
        run_name += "_FEWSHOT"
    if args.include_ref_range:
        if args.bin_lab_results:
            raise ValueError(
                "Binning and printing reference ranges concurrently is not supported."
            )
        run_name += "_REFRANGE"
    if args.only_abnormal_labs:
        run_name += "_ONLYABNORMAL"
    if args.bin_lab_results:
        run_name += "_BIN"
    if args.bin_lab_results_abnormal:
        run_name += "_BINABNORMAL"
    if not args.summarize:
        run_name += "_NOSUMMARY"
    if args.confirm_diagnosis:
        run_name += "_CONFIRM"
    if not args.abbreviated:
        run_name += "_NOABBR"
    if args.self_consistency:
        run_name += "_SELFCONSISTENCY"
    if prompt_template != FULL_INFO_TEMPLATE:
        run_name += f"_{args.prompt_template}"
    if args.save_probabilities:
        run_name += "_PROBS"
    if args.run_descr:
        run_name += str(args.run_descr)
    run_dir = join(args.local_logging_dir, run_name)

    os.makedirs(run_dir, exist_ok=True)

    # Setup logfile and logpickle
    results_log_path = join(run_dir, f"{run_name}_results.pkl")
    log_path = join(run_dir, f"{run_name}.log")
    logger.add(log_path, enqueue=True, backtrace=True, diagnose=True)
    logger.info(args)

    # Set langsmith project name
    # os.environ["LANGCHAIN_PROJECT"] = run_name

    # Load lab test mapping
    with open(args.lab_test_mapping_path, "rb") as f:
        lab_test_mapping_df = pickle.load(f)

    # Load patient data
    # for patho in ["appendicitis", "cholecystitis", "diverticulitis", "pancreatitis"]:
    patho = args.pathology
    hadm_info_clean = load_hadm_from_file(
        f"{patho}_hadm_info_first_diag", base_mimic=args.base_mimic
    )

    # Load list of specific IDs if provided
    patient_list = hadm_info_clean.keys()
    if args.patient_list_path:
        with open(args.patient_list_path, "rb") as f:
            patient_list = pickle.load(f)

    first_patient_seen = False
    for _id in patient_list:
        if args.first_patient and not first_patient_seen:
            if _id == args.first_patient:
                first_patient_seen = True
            else:
                continue
        logger.info(f"Processing patient: {_id}")
        hadm = hadm_info_clean[_id]

        # Fewshot
        fewshot_examples = ""
        if args.fewshot:
            if args.include_ref_range:
                fewshot_examples += FI_FEWSHOT_TEMPLATE_COPD_RR.format(
                    user_tag_start=tags["user_tag_start"],
                    user_tag_end=tags["user_tag_end"],
                    ai_tag_start=tags["ai_tag_start"],
                    ai_tag_end=tags["ai_tag_end"],
                )
                fewshot_examples += FI_FEWSHOT_TEMPLATE_PNEUMONIA_RR.format(
                    user_tag_start=tags["user_tag_start"],
                    user_tag_end=tags["user_tag_end"],
                    ai_tag_start=tags["ai_tag_start"],
                    ai_tag_end=tags["ai_tag_end"],
                )
            else:
                fewshot_examples += FI_FEWSHOT_TEMPLATE_COPD.format(
                    user_tag_start=tags["user_tag_start"],
                    user_tag_end=tags["user_tag_end"],
                    ai_tag_start=tags["ai_tag_start"],
                    ai_tag_end=tags["ai_tag_end"],
                )
                fewshot_examples += FI_FEWSHOT_TEMPLATE_PNEUMONIA.format(
                    user_tag_start=tags["user_tag_start"],
                    user_tag_end=tags["user_tag_end"],
                    ai_tag_start=tags["ai_tag_start"],
                    ai_tag_end=tags["ai_tag_end"],
                )

        # Diagnostic Criteria
        diagnostic_criteria = []
        if args.diagnostic_criteria:
            char_to_criteria = {
                "a": DIAGNOSTIC_CRITERIA_APPENDICITIS,
                "c": DIAGNOSTIC_CRITERIA_CHOLECYSTITIS,
                "d": DIAGNOSTIC_CRITERIA_DIVERTICULITIS,
                "p": DIAGNOSTIC_CRITERIA_PANCREATITIS,
            }
            for char in args.diagnostic_criteria:
                diagnostic_criteria.append(char_to_criteria[char])
        diagnostic_criteria = "\n".join(diagnostic_criteria)

        # Eval
        evaluator = load_evaluator(
            args.pathology
        )  # Reload every time to ensure no state is carried over

        input = ""
        rad_reports = ""

        input = add_patient_history(input, hadm, args.abbreviated)

        char_to_func = {
            "p": "include_physical_examination",
            "l": "include_laboratory_tests",
            "i": "include_imaging",
        }

        # Read desired order from mapping and args.order and then execute and parse result
        for char in args.order:
            func = char_to_func[char]
            # Must be within for loop to use updated input variable
            mapping_functions = {
                "include_imaging": (add_rad_reports, [input, hadm]),
                "include_physical_examination": (
                    add_physical_examination,
                    [input, hadm, args.abbreviated],
                ),
                "include_laboratory_tests": (
                    add_laboratory_tests,
                    [input, hadm, evaluator, lab_test_mapping_df, args],
                ),
            }

            function, input_params = mapping_functions[func]
            result = function(*input_params)

            if isinstance(result, tuple):
                input, rad_reports = result
            else:
                input = result

        # Escape previous curly brackets to avoid issues with format later
        input = input.replace("{", "{{").replace("}", "}}")
        # This we want to leave for future formatting
        input = input.replace("{{rad_reports}}", "{rad_reports}")

        input, fewshot_examples, rad_reports = control_context_length(
            input,
            prompt_template,
            fewshot_examples,
            args.include_ref_range,
            rad_reports,
            llm,
            args,
            tags,
            _id,
            hadm_info_clean,
            diagnostic_criteria,
            args.summarize,
        )

        result = chain.predict(
            input=input.format(rad_reports=rad_reports),
            fewshot_examples=fewshot_examples,
            diagnostic_criteria=diagnostic_criteria,
            stop=STOP_WORDS,
        )

        if args.prompt_template == "COT":
            # input = input.format(rad_reports=rad_reports)
            prompt_tokens_final_diag = calculate_num_tokens(
                llm.tokenizer,
                [
                    final_diagnosis_prompt.format(
                        # input=input,
                        cot=result,
                    )
                ],
            )
            if prompt_tokens_final_diag > args.max_context_length - 25:
                to_remove = prompt_tokens_final_diag - args.max_context_length + 25
                input = truncate_text(
                    llm.tokenizer,
                    # input,
                    -to_remove,
                )
                # input = input.replace("{", "{{").replace("}", "}}")

            result = final_diag_chain.predict(
                # input=input,
                cot=result,
                stop=STOP_WORDS,
            )

        # Confirm the diagnosis by providing the diagnostic criteria for the first answer and then asking the AI to confirm
        if args.confirm_diagnosis:
            # Remove numbering
            diagnosis = re.sub(r"\d+\.\s*", "", result)
            diagnosis = re.split(r"[,.\n]|(?:\s*\b(?:and|or|vs[.]?)\b\s*)", diagnosis)[
                0
            ]

            # If the length of the diagnosis is too long, the model didnt follow directions and we just take its answer
            if len(diagnosis.split()) > 10:
                append_to_pickle_file(results_log_path, {_id: result})
                continue

            DIAGNOSTIC_CRITERIA = read_dict(args.diagnostic_criteria_path)
            # Check if we have the diagnostic criteria for this pathology
            patho_match, score = process.extractOne(
                diagnosis, DIAGNOSTIC_CRITERIA.keys()
            )
            if score >= 80:
                patho = patho_match
                diagnostic_criteria = DIAGNOSTIC_CRITERIA.get(patho, None)
            else:
                # If not, let GPT write it for us
                diagnostic_criteria = write_diagnostic_criteria(
                    diagnosis, diag_crit_writer
                )
                DIAGNOSTIC_CRITERIA[diagnosis] = diagnostic_criteria
                write_dict(args.diagnostic_criteria_path, DIAGNOSTIC_CRITERIA)

            prompt_confirm = PromptTemplate(
                template=CONFIRM_DIAG_TEMPLATE,
                input_variables=["input", "result", "diagnostic_criteria"],
                partial_variables={
                    "system_tag_start": tags["system_tag_start"],
                    "system_tag_end": tags["system_tag_end"],
                    "user_tag_start": tags["user_tag_start"],
                    "user_tag_end": tags["user_tag_end"],
                    "ai_tag_start": tags["ai_tag_start"],
                    "ai_tag_end": tags["ai_tag_end"],
                },
            )
            chain_confirm = LLMChain(llm=llm, prompt=prompt_confirm)

            # Ensure we are not over our token limit, and if so, truncate
            input = input.format(rad_reports=rad_reports)
            prompt_tokens = calculate_num_tokens(
                llm.tokenizer,
                [
                    prompt_confirm.format(
                        input=input,
                        result=diagnosis,
                        diagnostic_criteria=diagnostic_criteria,
                    )
                ],
            )
            if prompt_tokens > args.max_context_length - 25:
                to_remove = prompt_tokens - args.max_context_length + 25
                input = truncate_text(
                    llm.tokenizer,
                    input,
                    -to_remove,
                )
                input = input.replace("{", "{{").replace("}", "}}")

            result = chain_confirm.predict(
                input=input,
                result=diagnosis,
                diagnostic_criteria=diagnostic_criteria,
                stop=STOP_WORDS,
            )

        if args.save_probabilities:
            append_to_pickle_file(
                results_log_path,
                {_id: {"Diagnosis": result, "Probabilities": llm.probabilities}},
            )
        else:
            append_to_pickle_file(results_log_path, {_id: result})


def write_diagnostic_criteria(pathology, diag_crit_writer):
    global STOP_WORDS
    diag_crit_prompt = PromptTemplate(
        template=WRITE_DIAG_CRITERIA_TEMPLATE,
        input_variables=["pathology"],
        partial_variables={
            "system_tag_start": gpt_tags["system_tag_start"],
            "system_tag_end": gpt_tags["system_tag_end"],
            "user_tag_start": gpt_tags["user_tag_start"],
            "user_tag_end": gpt_tags["user_tag_end"],
        },
    )
    diag_crit_chain = LLMChain(llm=diag_crit_writer, prompt=diag_crit_prompt)
    return diag_crit_chain.predict(
        pathology=pathology,
        stop=STOP_WORDS,
    )


def read_dict(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            fcntl.flock(file.fileno(), fcntl.LOCK_SH)
            data = json.load(file)
            fcntl.flock(file.fileno(), fcntl.LOCK_UN)
            return data
    return {}


def write_dict(file_path, data):
    with open(file_path, "w") as file:
        fcntl.flock(file.fileno(), fcntl.LOCK_EX)
        json.dump(data, file)
        fcntl.flock(file.fileno(), fcntl.LOCK_UN)


def add_patient_history(input, hadm, abbreviated=True):
    input += "@@@ PATIENT HISTORY @@@\n"
    # input += "PATIENT HISTORY\n"
    input += (
        hadm["Patient History"].strip()
        if abbreviated
        else hadm["Patient History Unabbreviated"].strip()
    )
    return input


def add_physical_examination(input, hadm, abbreviated=True):
    input += "\n\n@@@ PHYSICAL EXAMINATION @@@\n"
    # input += "\n\nPHYSICAL EXAMINATION\n"
    input += (
        hadm["Physical Examination"].strip()
        if abbreviated
        else hadm["Physical Examination Unabbreviated"].strip()
    )
    return input


def add_laboratory_tests(input, hadm, evaluator, lab_test_mapping_df, args):
    input += "\n\n@@@ LABORATORY RESULTS @@@\n"
    # input += "\n\nLABORATORY RESULTS\n"
    if args.include_ref_range:
        input += (
            "(<FLUID>) <TEST>: <RESULT> | REFERENCE RANGE (RR): [LOWER RR - UPPER RR]\n"
        )

    else:
        input += "(<FLUID>) <TEST>: <RESULT>\n"
    lab_tests_to_include = []

    for test_name in evaluator.required_lab_tests:
        lab_tests_to_include = (
            lab_tests_to_include + evaluator.required_lab_tests[test_name]
        )
    lab_tests_to_include = lab_tests_to_include + evaluator.neutral_lab_tests

    for test in lab_tests_to_include:
        if test in hadm["Laboratory Tests"].keys():
            input += create_lab_test_string(
                test,
                lab_test_mapping_df,
                hadm,
                include_ref_range=args.include_ref_range,
                bin_lab_results=args.bin_lab_results,
                bin_lab_results_abnormal=args.bin_lab_results_abnormal,
                only_abnormal_labs=args.only_abnormal_labs,
            )

    return input


def add_rad_reports(input, hadm):
    rad_reports = ""
    input += "\n\n@@@ IMAGING RESULTS @@@\n{rad_reports}"
    # input += "\n\nIMAGING RESULTS\n{rad_reports}"
    for rad in hadm["Radiology"]:
        if rad["Region"] == "Abdomen":
            rad_reports += f"\n{rad['Modality']} {rad['Region']}\n"
            rad_reports += f"{rad['Report']}".strip()
    return input, rad_reports


def control_context_length(
    input,
    prompt_template,
    fewshot_examples,
    include_ref_range,
    rad_reports,
    llm,
    args,
    tags,
    _id,
    hadm_info_clean,
    diagnostic_criteria,
    summarize,
):
    global STOP_WORDS
    max_context_length = args.max_context_length
    final_diagnosis_tokens = 25
    summarize_prompt = PromptTemplate(
        template=SUMMARIZE_OBSERVATION_TEMPLATE,
        input_variables=["observation"],
        partial_variables={
            "system_tag_start": tags["system_tag_start"],
            "system_tag_end": tags["system_tag_end"],
            "user_tag_start": tags["user_tag_start"],
            "user_tag_end": tags["user_tag_end"],
            "ai_tag_start": tags["ai_tag_start"],
        },
    )
    prompt_tokens = calculate_num_tokens(
        llm.tokenizer,
        [
            prompt_template.format(
                input=input.format(rad_reports=rad_reports),
                system_tag_start=tags["system_tag_start"],
                system_tag_end=tags["system_tag_end"],
                user_tag_start=tags["user_tag_start"],
                user_tag_end=tags["user_tag_end"],
                ai_tag_start=tags["ai_tag_start"],
                fewshot_examples=fewshot_examples,
                diagnostic_criteria=diagnostic_criteria,
            ),
        ],
    )
    # Check if our prompt would exceed the max context length and lead to truncation
    if prompt_tokens > max_context_length:
        # If fewshot can try taking some of the examples away
        if args.fewshot:
            # logger.warning(
            #    f"Patient {_id} has too long prompt. Attempting to remedy by removing a fewshot example."
            # )
            # Only include shorter sample i.e. COPD
            if include_ref_range:
                fewshot_examples = FI_FEWSHOT_TEMPLATE_COPD_RR.format(
                    user_tag_start=tags["user_tag_start"],
                    user_tag_end=tags["user_tag_end"],
                    ai_tag_start=tags["ai_tag_start"],
                    ai_tag_end=tags["ai_tag_end"],
                )
            else:
                fewshot_examples = FI_FEWSHOT_TEMPLATE_COPD.format(
                    user_tag_start=tags["user_tag_start"],
                    user_tag_end=tags["user_tag_end"],
                    ai_tag_start=tags["ai_tag_start"],
                    ai_tag_end=tags["ai_tag_end"],
                )
            prompt_tokens = calculate_num_tokens(
                llm.tokenizer,
                [
                    prompt_template.format(
                        input=input.format(rad_reports=rad_reports),
                        system_tag_start=tags["system_tag_start"],
                        system_tag_end=tags["system_tag_end"],
                        user_tag_start=tags["user_tag_start"],
                        user_tag_end=tags["user_tag_end"],
                        ai_tag_start=tags["ai_tag_start"],
                        fewshot_examples=fewshot_examples,
                        diagnostic_criteria=diagnostic_criteria,
                    ),
                ],
            )

            # If we're still too long, completely remove examples
            if prompt_tokens > max_context_length:
                # logger.warning(
                #    "Prompt is still too long. Removing all fewshot examples."
                # )
                fewshot_examples = ""
                prompt_tokens = calculate_num_tokens(
                    llm.tokenizer,
                    [
                        prompt_template.format(
                            input=input.format(rad_reports=rad_reports),
                            system_tag_start=tags["system_tag_start"],
                            system_tag_end=tags["system_tag_end"],
                            user_tag_start=tags["user_tag_start"],
                            user_tag_end=tags["user_tag_end"],
                            ai_tag_start=tags["ai_tag_start"],
                            fewshot_examples=fewshot_examples,
                            diagnostic_criteria=diagnostic_criteria,
                        ),
                    ],
                )

        # Before we start summarizing rad we should take a look if we are already over the context length
        prompt_tokens_no_rad = calculate_num_tokens(
            llm.tokenizer,
            [
                prompt_template.format(
                    input=input.format(rad_reports=""),
                    system_tag_start=tags["system_tag_start"],
                    system_tag_end=tags["system_tag_end"],
                    user_tag_start=tags["user_tag_start"],
                    user_tag_end=tags["user_tag_end"],
                    ai_tag_start=tags["ai_tag_start"],
                    fewshot_examples=fewshot_examples,
                    diagnostic_criteria=diagnostic_criteria,
                )
            ],
        )
        max_new_tokens = max_context_length - prompt_tokens_no_rad
        if max_new_tokens < final_diagnosis_tokens:
            # Even without rad, we are hitting our limit or close to. Need to remove rad and possibly truncate.
            rad_reports = ""
            # No rad and still too long, so truncate to max context length - final_diagnosis_tokens
            # logger.warning("Prompt is still too long. Truncating prompt.")
            to_truncate_length = (
                max_new_tokens
                - final_diagnosis_tokens  # Give a little wiggle room for the transitions and diagnosis
            )
            input = truncate_text(
                llm.tokenizer,
                input.format(rad_reports=rad_reports),
                to_truncate_length,
            )
            # Need to re-escape curly brackets
            input = input.replace("{", "{{").replace("}", "}}")
            return input, fewshot_examples, rad_reports

        # If we're still too long, then case is just longer than max context length and we need to summarize imaging results
        if prompt_tokens > max_context_length:
            if summarize:
                seen_modalities = set()
                rad_reports = ""
                # Go through original imaging and summarize
                for rad in hadm_info_clean[_id]["Radiology"]:
                    if (
                        rad["Region"] == "Abdomen"
                        and rad["Modality"] not in seen_modalities
                    ):
                        summarize_chain = LLMChain(llm=llm, prompt=summarize_prompt)
                        summary = summarize_chain.predict(
                            observation=rad["Report"], stop=STOP_WORDS
                        )
                        rad_reports += f"\n {summary}"
                        seen_modalities.add(rad["Modality"])
                prompt_tokens = calculate_num_tokens(
                    llm.tokenizer,
                    [
                        prompt_template.format(
                            input=input.format(rad_reports=rad_reports),
                            system_tag_start=tags["system_tag_start"],
                            system_tag_end=tags["system_tag_end"],
                            user_tag_start=tags["user_tag_start"],
                            user_tag_end=tags["user_tag_end"],
                            ai_tag_start=tags["ai_tag_start"],
                            fewshot_examples=fewshot_examples,
                            diagnostic_criteria=diagnostic_criteria,
                        ),
                    ],
                )

            # If we are still too long, summarize the summary and enforce max characters
            if prompt_tokens > max_context_length:
                if summarize:
                    summarize_chain = LLMChain(llm=llm, prompt=summarize_prompt)
                    # Make sure that the length of rad_reports summary prompt is less than max_context_length
                    prompt_tokens_summary = calculate_num_tokens(
                        llm.tokenizer,
                        [
                            summarize_prompt.format(
                                observation="",
                                system_tag_start=tags["system_tag_start"],
                                system_tag_end=tags["system_tag_end"],
                                user_tag_start=tags["user_tag_start"],
                                user_tag_end=tags["user_tag_end"],
                                ai_tag_start=tags["ai_tag_start"],
                            )
                        ],
                    )
                    prompt_tokens_rad = calculate_num_tokens(
                        llm.tokenizer,
                        [rad_reports],
                    )
                    if prompt_tokens_summary + prompt_tokens_rad > max_context_length:
                        rad_reports = truncate_text(
                            llm.tokenizer,
                            rad_reports,
                            max_context_length - prompt_tokens_summary - max_new_tokens,
                        )
                    rad_reports = summarize_chain.predict(
                        observation=rad_reports,
                        stop=STOP_WORDS,
                    )
                rad_reports = truncate_text(
                    llm.tokenizer,
                    rad_reports,
                    max_new_tokens - final_diagnosis_tokens,
                )  # give a little wiggle room for the transitions and diagnosis

    return input, fewshot_examples, rad_reports


if __name__ == "__main__":
    run()
