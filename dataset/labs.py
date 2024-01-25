import collections
from os.path import join
import os
import pandas as pd

from utils.nlp import extract_short_and_long_name
from tools.utils import (
    LAB_TEST_MAPPING_ALTERATIONS,
    ADDITIONAL_LAB_TEST_MAPPING,
    LAB_TEST_MAPPING_SYNONYMS,
    ADDITIONAL_LAB_TEST_MAPPING_SYNONYMS,
)


def parse_lab_events(lab_events_df_sf, _id):
    filtered_lab_events = lab_events_df_sf[lab_events_df_sf["hadm_id"] == _id]
    le, ref_r_low, ref_r_up = {}, {}, {}
    if not filtered_lab_events.empty:
        sorted_df = filtered_lab_events.sort_values(by="charttime", ascending=True)
        unique_lab_events_df = sorted_df.drop_duplicates(subset="itemid", keep="first")
        le = unique_lab_events_df.set_index("itemid")["valuestr"].to_dict()
        ref_r_low = unique_lab_events_df.set_index("itemid")[
            "ref_range_lower"
        ].to_dict()
        ref_r_up = unique_lab_events_df.set_index("itemid")["ref_range_upper"].to_dict()
    return le, ref_r_low, ref_r_up


def parse_microbio(microbio_df_sf, _id):
    filtered_microbio_df = microbio_df_sf[microbio_df_sf["hadm_id"] == _id]
    microbio = {}
    microbio_spec = {}

    # If there are multiple positive bacteria for an itemid, we want to merge them
    def return_value_string(group):
        first_row = group.iloc[0]
        if pd.isna(first_row.org_itemid):
            val_str = first_row.valuestr
        else:
            unique_org_itemid_values = group.dropna(subset=["org_itemid"])[
                "valuestr"
            ].unique()
            val_str = ", ".join(unique_org_itemid_values)
        return pd.Series([val_str, first_row.spec_itemid])

    if not filtered_microbio_df.empty:
        result = (
            filtered_microbio_df.groupby(["test_itemid", "charttime"])
            .apply(return_value_string)
            .reset_index()
        )
        result.columns = ["test_itemid", "charttime", "valuestr", "spec_itemid"]

        # Sort and drop duplicates, creating new DataFrames
        sorted_df = result.sort_values(by="charttime", ascending=True)
        unique_microbio_df = sorted_df.drop_duplicates(
            subset="test_itemid", keep="first"
        )
        microbio = unique_microbio_df.set_index("test_itemid")["valuestr"].to_dict()
        microbio_spec = unique_microbio_df.set_index("test_itemid")[
            "spec_itemid"
        ].to_dict()

    return microbio, microbio_spec


def find_and_append_abreviations(df):
    abbreviations = []
    for idx, row in df.iterrows():
        short_name, long_name = extract_short_and_long_name(row["label"])
        if short_name != long_name:
            abbreviations.append(
                (
                    row["label"],
                    short_name,
                    long_name,
                    row["corresponding_ids"],
                    row["fluid"],
                )
            )

    # convert abbreviations to dataframe
    df_new_entries = pd.DataFrame(
        abbreviations,
        columns=[
            "original_label",
            "short_name",
            "long_name",
            "corresponding_ids",
            "fluid",
        ],
    )

    # create new rows for both short and long name
    df_short = df_new_entries[["short_name", "corresponding_ids", "fluid"]].copy()
    df_short.rename(columns={"short_name": "label"}, inplace=True)

    df_long = df_new_entries[["long_name", "corresponding_ids", "fluid"]].copy()
    df_long.rename(columns={"long_name": "label"}, inplace=True)

    # append new entries to original dataframe
    df = pd.concat([df, df_short, df_long], ignore_index=True)

    return df


def create_corresponding_ids_from_duplicates(df):
    itemids_by_label = df.groupby("label")["itemid"].apply(list).to_dict()
    df["corresponding_ids"] = df["label"].map(itemids_by_label)
    return df


def fill_synonyms(df, pairs_dict):
    for key, val in pairs_dict.items():
        # Find corresponding_ids for key and val from the DataFrame
        key_ids = df.loc[df["itemid"] == key, "corresponding_ids"].values[0]
        val_ids = df.loc[df["itemid"] == val, "corresponding_ids"].values[0]

        # Merge and remove duplicates
        merged_ids = list(set(key_ids + val_ids))

        # Update corresponding_ids in DataFrame
        df.loc[df["itemid"] == key, "corresponding_ids"] = df.loc[
            df["itemid"] == key, "corresponding_ids"
        ].apply(lambda x: merged_ids)
        df.loc[df["itemid"] == val, "corresponding_ids"] = df.loc[
            df["itemid"] == val, "corresponding_ids"
        ].apply(lambda x: merged_ids)

    return df


def extend_corresponding_ids(df):
    item_dict = df.set_index("itemid")["corresponding_ids"].to_dict()
    item_dict = {k: v for k, v in item_dict.items() if not pd.isnull(k)}

    for itemid in item_dict.keys():
        queue = collections.deque([itemid])
        seen = set([itemid])

        while queue:
            cur_itemid = queue.popleft()
            for next_itemid in item_dict[cur_itemid]:
                if next_itemid not in seen:
                    seen.add(next_itemid)
                    queue.append(next_itemid)

        df.loc[df["itemid"] == itemid, "corresponding_ids"] = df.loc[
            df["itemid"] == itemid, "corresponding_ids"
        ].apply(lambda x: list(seen))

    return df


def prepend_total(df):
    total_entries = df[df["label"].str.contains("Total")][["itemid", "label"]]
    total_entries = total_entries[~total_entries["label"].str.startswith("Total")]

    corresponding_ids = total_entries["itemid"].values.tolist()
    corresponding_ids = [[c] for c in corresponding_ids]

    labels = total_entries["label"].values.tolist()
    labels = [label.replace(", Total", "") for label in labels]
    labels = [label.replace(" Total", "") for label in labels]
    labels = ["Total " + label for label in labels]

    total_df = pd.DataFrame({"label": labels, "corresponding_ids": corresponding_ids})
    df = pd.concat([df, total_df], ignore_index=True)
    return df


def generate_lab_test_mapping(
    base_mimic: str = "",
):
    # Create the mapping of possible tests to the actual test names
    base_hosp = join(base_mimic, "hosp")

    # We are only interested in those tests that have been performed at least once
    if os.path.exists(join(base_hosp, "d_labitems_min_1.csv")):
        lab_events_descr_df = pd.read_csv(join(base_hosp, "d_labitems_min_1.csv"))
    else:
        lab_description_df = pd.read_csv(join(base_mimic, "hosp", "d_labitems.csv"))
        lab_events_df = pd.read_csv(join(base_mimic, "hosp", "labevents.csv"))

        # first count the itemid in lab_events_df
        itemid_counts = lab_events_df["itemid"].value_counts().reset_index()
        itemid_counts.columns = ["itemid", "count"]

        # Then merge this with lab_descriptions_df
        lab_events_descr_df = pd.merge(
            lab_description_df, itemid_counts, on="itemid", how="left"
        )
        lab_events_descr_df["count"] = lab_events_descr_df["count"].fillna(0)

        lab_events_descr_df = lab_events_descr_df[lab_events_descr_df["count"] > 0]
        lab_events_descr_df.to_csv(join(base_hosp, "d_labitems_min_1.csv"), index=False)

    # Create the corresponding_ids column from each itemid and its duplicates
    lab_events_descr_df = create_corresponding_ids_from_duplicates(lab_events_descr_df)

    # Convert to Int64 to handle NaNs
    lab_events_descr_df["itemid"] = lab_events_descr_df["itemid"].astype("Int64")

    # Clean data so we don't interpret the words in the parantheses as abbreviations
    for to_replace, to_replace_with in LAB_TEST_MAPPING_ALTERATIONS.items():
        lab_events_descr_df["label"] = lab_events_descr_df["label"].replace(
            to_replace, to_replace_with
        )

    # Fill in synonyms
    lab_events_descr_df = fill_synonyms(lab_events_descr_df, LAB_TEST_MAPPING_SYNONYMS)

    for synonym, canonical in ADDITIONAL_LAB_TEST_MAPPING_SYNONYMS.items():
        ADDITIONAL_LAB_TEST_MAPPING[synonym] = ADDITIONAL_LAB_TEST_MAPPING[canonical]

    # Add custom mappings that have been encountered or medically verified
    for label, itemids in ADDITIONAL_LAB_TEST_MAPPING.items():
        lab_events_descr_df = pd.concat(
            [
                lab_events_descr_df,
                pd.DataFrame({"label": label, "corresponding_ids": [itemids]}),
            ],
            ignore_index=True,
        )

    # Check for names with abbreviations to be split and added to dataframe
    lab_events_descr_df = find_and_append_abreviations(lab_events_descr_df)

    # If name contains "Total", add version where "Total" is prefix
    lab_events_descr_df = prepend_total(lab_events_descr_df)

    # Propagete newly created
    lab_events_descr_df = extend_corresponding_ids(lab_events_descr_df)

    # Drop accidentally created duplicates
    lab_events_descr_df = lab_events_descr_df.drop_duplicates(
        subset=["itemid", "label", "fluid"]
    )

    # Import microbio events
    microbiology_df = pd.read_csv(join(base_hosp, "microbiologyevents.csv"))

    testid_to_name = microbiology_df.drop_duplicates("test_itemid").set_index(
        "test_itemid"
    )["test_name"]
    microbio_mapping_df = pd.DataFrame(
        list(testid_to_name.items()), columns=["itemid", "label"]
    )
    microbio_mapping_df = create_corresponding_ids_from_duplicates(microbio_mapping_df)
    lab_events_descr_df = pd.concat([lab_events_descr_df, microbio_mapping_df])

    # Convert corresponding_ids to int
    lab_events_descr_df["corresponding_ids"] = lab_events_descr_df[
        "corresponding_ids"
    ].apply(lambda x: [int(y) for y in x])

    # save mapping as pickle
    lab_events_descr_df.to_pickle(join(base_hosp, "lab_test_mapping.pkl"))
