import pandas as pd
import os
import csv

def load_vn_mappings(train_csv_path, val_csv_path):
    """Creates mapping dictionaries: verb_id → verb_name, noun_id → noun_name."""
    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)

    # Combine and drop duplicates
    combined_df = pd.concat([train_df, val_df], ignore_index=True)
    verb_mapping = dict(zip(combined_df['verb_class'], combined_df['verb']))
    noun_mapping = dict(zip(combined_df['noun_class'], combined_df['noun']))

    # In case there are duplicates, keep only the first
    verb_mapping = {k: v for k, v in sorted(verb_mapping.items())}
    noun_mapping = {k: v for k, v in sorted(noun_mapping.items())}

    return verb_mapping, noun_mapping


# This function has being taken with minor changes from:
# https://github.com/zhaoyue-zephyrus/AVION/blob/main/avion/utils/misc.py
def generate_label_map(train_csv, val_csv):
    print("Preprocessing EK100 action label space...")
    vn_list = []
    mapping_vn2narration = {}
    for f in [train_csv, val_csv]:
        csv_reader = csv.reader(open(f))
        _ = next(csv_reader)  # skip header
        for row in csv_reader:
            vn = '{}:{}'.format(int(row[10]), int(row[12]))  # verb_id:noun_id
            narration = row[8]  
            if vn not in vn_list:
                vn_list.append(vn)
            if vn not in mapping_vn2narration:
                mapping_vn2narration[vn] = [narration]
            else:
                mapping_vn2narration[vn].append(narration)

    vn_list = sorted(vn_list)  
    mapping_vn2act = {vn: i for i, vn in enumerate(vn_list)}  
    print(f"Number of actions = {len(mapping_vn2act)}")
    return mapping_vn2act
