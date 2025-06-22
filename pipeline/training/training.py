import json
import os
import torch
import random
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import (GATConv, TransformerConv, HeteroConv, global_mean_pool)
from torch_geometric.loader import DataLoader
from omegaconf import OmegaConf
import wandb
import time
import numpy as np
import argparse
from sklearn.metrics import confusion_matrix, top_k_accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

from tgnn_model import GraphClassifier
from data_loader import GraphDataset
from vn_mappings import generate_label_map, load_vn_mappings
from tqdm import tqdm




def plot_confusion_matrix(true_labels, pred_labels, label_type="Verb"):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, cmap="Blues", fmt="d")
    plt.title(f"{label_type} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


def log_confusion_matrix(true_labels, pred_labels, class_names, label_type="Verb", epoch=0):
    """Logs a normalized confusion matrix to WandB."""
    cm = confusion_matrix(true_labels, pred_labels)

    with np.errstate(all='ignore'):
        cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm_normalized = np.nan_to_num(cm_normalized)  # replace NaN with 0

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=False, cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Proportion'})
    plt.title(f"{label_type} Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    wandb.log({f"{label_type}_Confusion_Matrix": wandb.Image(fig)}, step=epoch)
    plt.close(fig)


def train_fn(model, train_loader, optimizer, criterion, device, mapping_vn2act):
    model.train()
    total_loss, total_samples = 0, 0


    mapping_act2v = {i: int(vn.split(':')[0]) for (vn, i) in mapping_vn2act.items()}
    mapping_act2n = {i: int(vn.split(':')[1]) for (vn, i) in mapping_vn2act.items()}


    all_preds = []
    all_targets = []

    pred_verb_ids = []
    pred_noun_ids = []
    true_verb_ids = []
    true_noun_ids = []

    for batch in tqdm(train_loader, desc="Training"):
        batch = batch.to(device)
        optimizer.zero_grad()

        # for key in batch.x_dict.keys():
        #     if 'object' in key:
        #         noise_std = 0.5
        #         batch.x_dict[key] = batch.x_dict[key] + torch.randn_like(batch.x_dict[key]) * noise_std


        out = model(batch.x_dict, batch.edge_index_dict, 
                    {'relation': batch['object', 'relation', 'object'].edge_attr}, 
                    batch)

        loss = criterion(out, batch.y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_samples += batch.y.size(0)

        # Compute and print the maximum gradient norm for the model parameters
        # max_grad_norm = 0
        # total_grad_norm = 0
        # count_grad = 0
        # for param in model.parameters():
        #     if param.grad is not None:
        #         grad_norm = param.grad.data.norm(2).item()
        #         total_grad_norm += grad_norm
        #         count_grad += 1
        #         if grad_norm > max_grad_norm:
        #             max_grad_norm = grad_norm
        # mean_grad_norm = total_grad_norm / count_grad if count_grad > 0 else 0
        # print(f"Max gradient norm: {max_grad_norm:.4f} | Mean gradient norm: {mean_grad_norm:.4f}")

        # Decode verbs and nouns
        preds = out.argmax(dim=1)
        
        # out is of shape (N, num_classes)
        # batch.y is of shape (N, num_classes)

        # add to all pred individual predictions
        for i in range(out.shape[0]):
            all_preds.append(out[i].detach().cpu())
            all_targets.append(torch.argmax(batch.y[i].detach().cpu()))



        for pred_idx, true_idx in zip(preds.cpu().numpy(), batch.y.cpu().numpy()):
            true_idx = true_idx.argmax()
            pred_verb = mapping_act2v[pred_idx]
            pred_noun = mapping_act2n[pred_idx]
            true_verb = mapping_act2v[true_idx]
            true_noun = mapping_act2n[true_idx]

            pred_verb_ids.append(pred_verb)
            pred_noun_ids.append(pred_noun)
            true_verb_ids.append(true_verb)
            true_noun_ids.append(true_noun)

    # === Compute metrics ===


    acc1 = top_k_accuracy_score(
        np.array(all_targets), 
        np.array(all_preds), 
        k=1, 
        labels=np.arange(len(mapping_vn2act))
    )
    acc5 = top_k_accuracy_score(
        np.array(all_targets), 
        np.array(all_preds), 
        k=5, 
        labels=np.arange(len(mapping_vn2act))
    )

    pred_verb_ids = np.array(pred_verb_ids)
    pred_noun_ids = np.array(pred_noun_ids)
    true_verb_ids = np.array(true_verb_ids)
    true_noun_ids = np.array(true_noun_ids)

    verb_acc = (pred_verb_ids == true_verb_ids).sum() / len(true_verb_ids)
    noun_acc = (pred_noun_ids == true_noun_ids).sum() / len(true_noun_ids)

    avg_loss = total_loss / len(train_loader)

    return avg_loss, acc1, acc5, verb_acc, noun_acc


def eval_fn(model, val_loader, criterion, device, mapping_vn2act):
    model.eval()
    total_loss = 0

    mapping_act2v = {i: int(vn.split(':')[0]) for (vn, i) in mapping_vn2act.items()}
    mapping_act2n = {i: int(vn.split(':')[1]) for (vn, i) in mapping_vn2act.items()}

    all_preds = []
    all_targets = []

    pred_verb_ids = []
    pred_noun_ids = []
    true_verb_ids = []
    true_noun_ids = []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out = model(
                batch.x_dict,
                batch.edge_index_dict,
                {'relation': batch['object', 'relation', 'object'].edge_attr},
                batch
            )

            loss = criterion(out, batch.y)
            total_loss += loss.item()

            # Collect individual predictions and targets, as in train_fn.
            for i in range(out.shape[0]):
                all_preds.append(out[i].detach().cpu())
                all_targets.append(torch.argmax(batch.y[i].detach().cpu()))

            preds = out.argmax(dim=1)
            for pred_idx, true_vec in zip(preds.cpu().numpy(), batch.y.cpu().numpy()):
                true_idx = true_vec.argmax()
                # print("True idx: ", true_idx)
                # print("Pred idx", pred_idx)
                pred_verb = mapping_act2v[pred_idx]
                pred_noun = mapping_act2n[pred_idx]
                true_verb = mapping_act2v[true_idx]
                true_noun = mapping_act2n[true_idx]
                pred_verb_ids.append(pred_verb)
                pred_noun_ids.append(pred_noun)
                true_verb_ids.append(true_verb)
                true_noun_ids.append(true_noun)

    all_preds = torch.stack(all_preds).numpy()
    all_targets = np.array(all_targets)

    acc1 = top_k_accuracy_score(
        all_targets, 
        all_preds, 
        k=1, 
        labels=np.arange(len(mapping_vn2act))
    )
    acc5 = top_k_accuracy_score(
        all_targets, 
        all_preds, 
        k=5, 
        labels=np.arange(len(mapping_vn2act))
    )

    verb_acc = (np.array(pred_verb_ids) == np.array(true_verb_ids)).sum() / len(true_verb_ids)
    noun_acc = (np.array(pred_noun_ids) == np.array(true_noun_ids)).sum() / len(true_noun_ids)
    avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0

    return avg_loss, acc1, acc5, verb_acc, noun_acc, true_verb_ids, pred_verb_ids, true_noun_ids, pred_noun_ids


def main(args):
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")      
    embedder = None

    DATASET_DIR = args.dataset_dir
    graph_dir_train = os.path.join(DATASET_DIR, "train")
    graph_dir_val = os.path.join(DATASET_DIR, "val")

    train_csv = "conf/ek100/EPIC_100_train.csv"
    val_csv = "conf/ek100/EPIC_100_train.csv"

    label, mapping_vn2act = generate_label_map('ek100_cls')
    mapping_act2v = {i: int(vn.split(':')[0]) for (vn, i) in mapping_vn2act.items()}
    mapping_act2n = {i: int(vn.split(':')[1]) for (vn, i) in mapping_vn2act.items()}


    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LR = args.lr

    print(f"Num actions: {len(mapping_vn2act)}")

    model = GraphClassifier(3072, 128, 4, len(mapping_vn2act)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    train_dataset = GraphDataset(
        data_dir=args.dataset_dir,
        embedder=embedder,
        metadata_csv=train_csv,
        mapping_vn2act=mapping_vn2act,
        node_drop_p=0.2,
        is_train=True
    )
    val_dataset = GraphDataset(
        data_dir=args.dataset_dir,
        embedder=embedder,
        metadata_csv=train_csv,
        mapping_vn2act=mapping_vn2act,
        node_drop_p=0.0,
        is_train=False
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    wandb.init(project="ar_graph", 
            name=f"tgnn_{time.time()}", 
            config={
                "learning_rate": LR, 
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "dataset": "EPIC-Kitchens-100",
                "n_classes": len(mapping_vn2act)}
    )

    os.makedirs("logs/", exist_ok=True)
    model_output = os.path.join("logs/", 'best_model.pt')

    best_valid_loss = np.Inf
    since = time.time()

    for epoch in range(EPOCHS):
        avg_train_loss, train_acc1, train_acc5, train_verb_acc, train_noun_acc = train_fn(
            model, train_loader, optimizer, criterion, device, mapping_vn2act)
        avg_val_loss, val_acc1, val_acc5, val_verb_acc, val_noun_acc, true_verb_ids, pred_verb_ids, true_noun_ids, pred_noun_ids = eval_fn(
            model, val_loader, criterion, device, mapping_vn2act)

        # if avg_val_loss < best_valid_loss:
        #     torch.save(model.state_dict(), model_output)
        #     best_valid_loss = avg_val_loss
        #     print("SAVED_WEIGHTS_SUCCESS")

        wandb.log({
            "train/loss": avg_train_loss,
            "train/acc@1": train_acc1,
            "train/acc@5": train_acc5,
            "train/verb_acc@1": train_verb_acc,
            "train/noun_acc@1": train_noun_acc,
            "val/loss": avg_val_loss,
            "val/acc@1": val_acc1,
            "val/acc@5": val_acc5,
            "val/verb_acc@1": val_verb_acc,
            "val/noun_acc@1": val_noun_acc
        }, step=epoch)

        print(f"Epoch {epoch+1}: " , f"TRAIN Loss {avg_train_loss:.3f} | Acc@1 {train_acc1:.3f} | Acc@5 {train_acc5:.3f} | V.Acc {train_verb_acc:.3f} | N.Acc {train_noun_acc:.3f} VAL Loss {avg_val_loss:.3f} | Acc@1 {val_acc1:.3f} | Acc@5 {val_acc5:.3f} | Verb Acc {val_verb_acc:.3f} | N.Acc {val_noun_acc:.3f}")

        # Log confusion matrices
        # verb_class_names = [verb_mapping[v] if v in verb_mapping else str(v) for v in np.unique(true_verb_ids)]
        # noun_class_names = [noun_mapping[n] if n in noun_mapping else str(n) for n in np.unique(true_noun_ids)]
        # log_confusion_matrix(true_verb_ids, pred_verb_ids, verb_class_names, label_type="Verb", epoch=epoch)
        # log_confusion_matrix(true_noun_ids, pred_noun_ids, noun_class_names, label_type="Noun", epoch=epoch)

        # Save last model
        torch.save(model.state_dict(), os.path.join("logs", 'last_model.pt'))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    wandb.finish()


if __name__ == "__main__":    
    argparser = argparse.ArgumentParser(description="Train a Temporal Graph Neural Network")
    argparser.add_argument("--dataset_dir", type=str,
                           default="datasets/EK100/tests/ar_graph", 
                           help="Path to the graph dataset directory")
    argparser.add_argument("--train_csv", type=str, 
                           default="datasets/EK100/epic-kitchens-100-annotations/EPIC_100_train.csv",
                           help="Path to the training CSV file")
    argparser.add_argument("--val_csv", type=str, 
                            default="tasets/EK100/epic-kitchens-100-annotations/EPIC_100_val.csv",
                           help="Path to the validation CSV file")
    argparser.add_argument("--batch_size", type=int, default=16, 
                           help="Batch size for training")
    argparser.add_argument("--epochs", type=int, default=300, 
                           help="Number of epochs for training")
    argparser.add_argument("--lr", type=float, default=0.0001, 
                           help="Learning rate for the optimizer")
    argparser.add_argument("--model_output_dir", type=str, 
                           default="trained_models", 
                           help="Directory to save the trained model")
    argparser.add_argument("--seed", type=int, default=42,
                           help="Random seed for reproducibility")

    args = argparser.parse_args()
    main(args)
