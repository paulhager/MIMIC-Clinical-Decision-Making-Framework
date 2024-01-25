import os
from os.path import join
import random
from datetime import datetime
import time
import pickle

import numpy as np
import hydra
from omegaconf import DictConfig
from loguru import logger
import langchain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from dataset.utils import load_hadm_from_file
from utils.logging import append_to_pickle_file
from utils.nlp import create_lab_test_string
from models.models import CustomLLM

from agents.prompts import REFERENCE_RANGE_TEST_RR, REFERENCE_RANGE_TEST_ZEROSHOT


@hydra.main(config_path="./configs", config_name="config", version_base=None)
def run(args: DictConfig):
    random.seed(args.seed)
    np.random.seed(args.seed)

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
    )
    llm.load_model(args.base_models)

    if args.fewshot:
        prompt = PromptTemplate(
            template=REFERENCE_RANGE_TEST_RR,
            input_variables=["lab_test_string_rr"],
            partial_variables={
                "system_tag_start": tags["system_tag_start"],
                "system_tag_end": tags["system_tag_end"],
                "user_tag_start": tags["user_tag_start"],
                "user_tag_end": tags["user_tag_end"],
                "ai_tag_start": tags["ai_tag_start"],
                "ai_tag_end": tags["ai_tag_end"],
            },
        )
    else:
        prompt = PromptTemplate(
            template=REFERENCE_RANGE_TEST_ZEROSHOT,
            input_variables=["lab_test_string_rr"],
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
    run_name = f"{args.model_name}_{str_date}_RRTEST"

    if args.fewshot:
        run_name += "_FEWSHOT"
    if args.run_descr:
        run_name += str(args.run_descr)
    run_dir = join(args.local_logging_dir, run_name)

    os.makedirs(run_dir, exist_ok=True)

    # Setup logfile and logpickle
    results_log_path = join(run_dir, f"{run_name}_results.pkl")
    log_path = join(run_dir, f"{run_name}.log")
    logger.add(log_path, enqueue=True, backtrace=True, diagnose=True)
    logger.info(args)

    # Load lab test mapping
    with open(args.lab_test_mapping_path, "rb") as f:
        lab_test_mapping_df = pickle.load(f)

    for pathology in [
        "appendicitis",
        "cholecystitis",
        "diverticulitis",
        "pancreatitis",
    ]:
        # Load patient data
        hadm_info_clean = load_hadm_from_file(
            f"{pathology}_hadm_info_first_diag", base_mimic=args.base_mimic
        )
        for _id in hadm_info_clean:
            logger.info(f"Processing patient: {_id}")
            hadm_info = hadm_info_clean[_id]

            results = []
            for test, value in hadm_info["Laboratory Tests"].items():
                rr_lower = hadm_info["Reference Range Lower"][test]
                rr_upper = hadm_info["Reference Range Upper"][test]
                if value[0].isdigit() and rr_lower == rr_lower and rr_upper == rr_upper:
                    lab_test_string_rr = create_lab_test_string(
                        test, lab_test_mapping_df, hadm_info, include_ref_range=True
                    ).strip()
                    gt = (
                        create_lab_test_string(
                            test, lab_test_mapping_df, hadm_info, bin_lab_results=True
                        )
                        .strip()
                        .split()[-1]
                    )
                    result = chain.predict(
                        input=input,
                        lab_test_string_rr=lab_test_string_rr,
                        stop=[],
                    )
                    result = result.split()
                    if len(result) == 0:
                        result = ""
                    else:
                        result = result[0]
                    results.append(
                        {
                            "ID": _id,
                            "Test": test,
                            "Value": value,
                            "Lower RR": rr_lower,
                            "Upper RR": rr_upper,
                            "Result": result,
                            "GT": gt,
                            "Pathology": pathology,
                        }
                    )
            append_to_pickle_file(results_log_path, results)


if __name__ == "__main__":
    run()
