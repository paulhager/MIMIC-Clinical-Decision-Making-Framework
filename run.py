import os
from os.path import join
import random
from datetime import datetime
import time

import numpy as np
import torch

import hydra
from omegaconf import DictConfig
from loguru import logger
import langchain

from dataset.utils import load_hadm_from_file
from utils.logging import append_to_pickle_file
from evaluators.appendicitis_evaluator import AppendicitisEvaluator
from evaluators.cholecystitis_evaluator import CholecystitisEvaluator
from evaluators.diverticulitis_evaluator import DiverticulitisEvaluator
from evaluators.pancreatitis_evaluator import PancreatitisEvaluator
from models.models import CustomLLM
from agents.agent import build_agent_executor_ZeroShot


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
    if not args.self_consistency:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    # Load patient data
    hadm_info_clean = load_hadm_from_file(
        f"{args.pathology}_hadm_info_first_diag", base_mimic=args.base_mimic
    )

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

    date_time = datetime.fromtimestamp(time.time())
    str_date = date_time.strftime("%d-%m-%Y_%H:%M:%S")
    args.model_name = args.model_name.replace("/", "_")
    run_name = f"{args.pathology}_{args.agent}_{args.model_name}_{str_date}"
    if args.fewshot:
        run_name += "_FEWSHOT"
    if args.include_ref_range:
        if args.bin_lab_results:
            raise ValueError(
                "Binning and printing reference ranges concurrently is not supported."
            )
        run_name += "_REFRANGE"
    if args.bin_lab_results:
        run_name += "_BIN"
    if args.include_tool_use_examples:
        run_name += "_TOOLEXAMPLES"
    if args.provide_diagnostic_criteria:
        run_name += "_DIAGCRIT"
    if not args.summarize:
        run_name += "_NOSUMMARY"
    if args.run_descr:
        run_name += str(args.run_descr)
    run_dir = join(args.local_logging_dir, run_name)

    os.makedirs(run_dir, exist_ok=True)

    # Setup logfile and logpickle
    results_log_path = join(run_dir, f"{run_name}_results.pkl")
    eval_log_path = join(run_dir, f"{run_name}_eval.pkl")
    log_path = join(run_dir, f"{run_name}.log")
    logger.add(log_path, enqueue=True, backtrace=True, diagnose=True)
    langchain.debug = True

    # Set langsmith project name
    # os.environ["LANGCHAIN_PROJECT"] = run_name

    # Predict for all patients
    first_patient_seen = False
    for _id in hadm_info_clean.keys():
        if args.first_patient and not first_patient_seen:
            if _id == args.first_patient:
                first_patient_seen = True
            else:
                continue

        logger.info(f"Processing patient: {_id}")

        # Build
        agent_executor = build_agent_executor_ZeroShot(
            patient=hadm_info_clean[_id],
            llm=llm,
            lab_test_mapping_path=args.lab_test_mapping_path,
            logfile=log_path,
            max_context_length=args.max_context_length,
            tags=tags,
            include_ref_range=args.include_ref_range,
            bin_lab_results=args.bin_lab_results,
            include_tool_use_examples=args.include_tool_use_examples,
            provide_diagnostic_criteria=args.provide_diagnostic_criteria,
            summarize=args.summarize,
            model_stop_words=args.stop_words,
        )

        # Run
        result = agent_executor(
            {"input": hadm_info_clean[_id]["Patient History"].strip()}
        )
        append_to_pickle_file(results_log_path, {_id: result})


if __name__ == "__main__":
    run()
