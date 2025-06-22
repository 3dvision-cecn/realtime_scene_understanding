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
def generate_label_map(dataset):
    if dataset == 'ek100_cls':
        print("Preprocess ek100 action label space")
        vn_list = []
        mapping_vn2narration = {}
        for f in [
            'conf/ek100/EPIC_100_train.csv',
            'conf/ek100/EPIC_100_validation.csv',
        ]:
            csv_reader = csv.reader(open(f))
            _ = next(csv_reader)  # skip the header
            for row in csv_reader:
                vn = '{}:{}'.format(int(row[10]), int(row[12]))
                narration = row[8]
                if vn not in vn_list:
                    vn_list.append(vn)
                if vn not in mapping_vn2narration:
                    mapping_vn2narration[vn] = [narration]
                else:
                    mapping_vn2narration[vn].append(narration)
                # mapping_vn2narration[vn] = [narration]
        vn_list = sorted(vn_list)
        print('# of action= {}'.format(len(vn_list)))
        mapping_vn2act = {vn: i for i, vn in enumerate(vn_list)}
        labels = [list(set(mapping_vn2narration[vn_list[i]])) for i in range(len(mapping_vn2act))]
        # shape of the labels
        # print(len(labels), len(labels[0]), labels[0])
        # print(labels[:5])
    elif dataset == 'charades_ego':
        print("=> preprocessing charades_ego action label space")
        vn_list = []
        labels = []
        with open('datasets/CharadesEgo/CharadesEgo/Charades_v1_classes.txt') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                vn = row[0][:4]
                vn_list.append(vn)
                narration = row[0][5:]
                labels.append(narration)
        mapping_vn2act = {vn: i for i, vn in enumerate(vn_list)}
    elif dataset == 'egtea':
        print("=> preprocessing egtea action label space")
        labels = []
        with open('datasets/EGTEA/action_idx.txt') as f:
            for row in f:
                row = row.strip()
                narration = ' '.join(row.split(' ')[:-1])
                labels.append(narration.replace('_', ' ').lower())
                # labels.append(narration)
        mapping_vn2act = {label: i for i, label in enumerate(labels)}
    else:
        raise NotImplementedError
    return labels, mapping_vn2act