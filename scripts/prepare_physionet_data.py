#!/usr/bin/env python3
"""Convert the released MIMIC-CDM CSV files to the framework's runtime format."""

import argparse
import ast
import json
import pickle
from pathlib import Path

import pandas as pd


PATHOLOGIES = (
    "appendicitis",
    "cholecystitis",
    "diverticulitis",
    "pancreatitis",
)


def grouped_values(frame: pd.DataFrame, column: str) -> dict[int, list]:
    return (
        frame.dropna(subset=[column])
        .groupby("hadm_id", sort=False)[column]
        .apply(list)
        .to_dict()
    )


def load_released_dataset(source: Path) -> tuple[dict, pd.DataFrame]:
    pathology_ids = json.loads((source / "pathology_ids.json").read_text())
    hpi = pd.read_csv(source / "history_of_present_illness.csv").set_index("hadm_id")
    physical = pd.read_csv(source / "physical_examination.csv").set_index("hadm_id")
    discharge = pd.read_csv(source / "discharge_diagnosis.csv").set_index("hadm_id")
    discharge_procedures = grouped_values(
        pd.read_csv(source / "discharge_procedures.csv"),
        "discharge_procedure",
    )
    icd_diagnoses = grouped_values(
        pd.read_csv(source / "icd_diagnosis.csv"),
        "icd_diagnosis",
    )

    labs = pd.read_csv(source / "laboratory_tests.csv")
    labs["itemid"] = labs["itemid"].astype(int)
    labs_by_hadm = {
        int(hadm_id): group for hadm_id, group in labs.groupby("hadm_id", sort=False)
    }

    microbiology = pd.read_csv(source / "microbiology.csv")
    microbiology["test_itemid"] = pd.to_numeric(
        microbiology["test_itemid"], errors="coerce"
    )
    microbiology = microbiology.dropna(subset=["test_itemid"])
    microbiology["test_itemid"] = microbiology["test_itemid"].astype(int)
    microbiology_by_hadm = {
        int(hadm_id): group
        for hadm_id, group in microbiology.groupby("hadm_id", sort=False)
    }

    radiology = pd.read_csv(source / "radiology_reports.csv")
    radiology_by_hadm = {
        int(hadm_id): group
        for hadm_id, group in radiology.groupby("hadm_id", sort=False)
    }

    icd_procedures = pd.read_csv(
        source / "icd_procedures.csv",
        dtype={"icd_code": str},
    )
    icd_procedures["icd_version"] = icd_procedures["icd_version"].astype(int)
    procedures_by_hadm = {
        int(hadm_id): group
        for hadm_id, group in icd_procedures.groupby("hadm_id", sort=False)
    }

    patients = {}
    for hadm_id in hpi.index:
        hadm_id = int(hadm_id)
        lab_rows = labs_by_hadm.get(hadm_id)
        micro_rows = microbiology_by_hadm.get(hadm_id)
        rad_rows = radiology_by_hadm.get(hadm_id)
        procedure_rows = procedures_by_hadm.get(hadm_id)

        patient = {
            "Patient History": str(hpi.at[hadm_id, "hpi"]),
            "Patient History Unabbreviated": str(hpi.at[hadm_id, "hpi"]),
            "Physical Examination": str(physical.at[hadm_id, "pe"]),
            "Physical Examination Unabbreviated": str(physical.at[hadm_id, "pe"]),
            "Laboratory Tests": {},
            "Reference Range Lower": {},
            "Reference Range Upper": {},
            "Microbiology": {},
            "Radiology": [],
            "Discharge Diagnosis": str(
                discharge.at[hadm_id, "discharge_diagnosis"]
            ),
            "ICD Diagnosis": icd_diagnoses.get(hadm_id, []),
            "Procedures": discharge_procedures.get(hadm_id, []),
            "Procedures ICD9": [],
            "Procedures ICD9 Title": [],
            "Procedures ICD10": [],
            "Procedures ICD10 Title": [],
        }

        if lab_rows is not None:
            patient["Laboratory Tests"] = dict(
                zip(lab_rows["itemid"], lab_rows["valuestr"])
            )
            patient["Reference Range Lower"] = dict(
                zip(lab_rows["itemid"], lab_rows["ref_range_lower"])
            )
            patient["Reference Range Upper"] = dict(
                zip(lab_rows["itemid"], lab_rows["ref_range_upper"])
            )

        if micro_rows is not None:
            patient["Microbiology"] = dict(
                zip(micro_rows["test_itemid"], micro_rows["valuestr"])
            )

        if rad_rows is not None:
            patient["Radiology"] = [
                {
                    "Report": str(row.text),
                    "Modality": str(row.modality),
                    "Region": str(row.region),
                }
                for row in rad_rows.itertuples()
            ]

        if procedure_rows is not None:
            for row in procedure_rows.itertuples():
                if row.icd_version == 9:
                    patient["Procedures ICD9"].append(int(row.icd_code))
                    patient["Procedures ICD9 Title"].append(row.icd_title)
                else:
                    patient["Procedures ICD10"].append(row.icd_code)
                    patient["Procedures ICD10 Title"].append(row.icd_title)

        patients[hadm_id] = patient

    mapping = pd.read_csv(source / "lab_test_mapping.csv")
    mapping["corresponding_ids"] = mapping["corresponding_ids"].map(
        ast.literal_eval
    )
    return {"patients": patients, "pathology_ids": pathology_ids}, mapping


def write_runtime_data(source: Path, output: Path) -> dict[str, int]:
    loaded, mapping = load_released_dataset(source)
    patients = loaded["patients"]
    pathology_ids = loaded["pathology_ids"]
    output.mkdir(parents=True, exist_ok=True)

    counts = {}
    for pathology in PATHOLOGIES:
        selected = {
            int(hadm_id): patients[int(hadm_id)]
            for hadm_id in pathology_ids[pathology]
        }
        destination = output / f"{pathology}_hadm_info_first_diag.pkl"
        with destination.open("wb") as output_file:
            pickle.dump(selected, output_file)
        with (output / f"{pathology}_smoke_patient.pkl").open("wb") as output_file:
            pickle.dump([next(iter(selected))], output_file)
        counts[pathology] = len(selected)

    mapping.to_pickle(output / "lab_test_mapping.pkl")
    (output / "runs").mkdir(exist_ok=True)
    return counts


def write_local_paths(repo_root: Path, runtime_data: Path) -> Path:
    destination = repo_root / "configs" / "paths" / "local.yaml"
    destination.write_text(
        "\n".join(
            [
                "# @package _global_",
                "defaults:",
                "  - _self_",
                "",
                f'base_mimic: "{runtime_data}"',
                'base_models: ""',
                f'lab_test_mapping_path: "{runtime_data / "lab_test_mapping.pkl"}"',
                f'local_logging_dir: "{runtime_data / "runs"}"',
                "",
            ]
        )
    )
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source",
        type=Path,
        help="Directory containing the credentialed MIMIC-CDM release CSV files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "runtime_data",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    required = {
        "pathology_ids.json",
        "history_of_present_illness.csv",
        "physical_examination.csv",
        "laboratory_tests.csv",
        "lab_test_mapping.csv",
    }
    missing = sorted(name for name in required if not (source / name).is_file())
    if missing:
        raise SystemExit(f"Missing required files: {', '.join(missing)}")

    counts = write_runtime_data(source, output)
    paths_file = write_local_paths(Path(__file__).resolve().parents[1], output)
    print(f"runtime data: {output}")
    print(f"paths config: {paths_file}")
    print("cases: " + ", ".join(f"{name}={count}" for name, count in counts.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
