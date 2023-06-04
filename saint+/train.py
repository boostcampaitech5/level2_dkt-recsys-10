import time
import pickle
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from args import parser
from model import SaintPlus, NoamOpt
from torch.utils.data import DataLoader
from data_generator import Riiid_Sequence
from sklearn.metrics import roc_auc_score
import os

# 모델 저장
def save_model(model):
    check_point = {
        'net': model.state_dict()
    }
    torch.save(check_point,"./model/fianlSaint+.pt")

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    args = parser.parse_args()
    num_layers = args.num_layers
    num_heads = args.num_heads
    d_model = args.d_model
    d_ffn = d_model*4
    max_len = args.max_len
    n_questions = args.n_questions
    n_parts = args.n_parts
    n_tasks = args.n_tasks
    n_ans = args.n_ans

    seq_len = args.seq_len
    warmup_steps = args.warmup_steps
    dropout = args.dropout
    epochs = args.epochs
    patience = args.patience
    batch_size = args.batch_size
    
    # train, valid 데이터 가져오기
    with open("./after_data/train_group90.pkl.zip", 'rb') as pick:
        train_group = pickle.load(pick)
    with open("./after_data/val_group90.pkl.zip", 'rb') as pick:
        val_group = pickle.load(pick)

    train_seq = Riiid_Sequence(train_group, seq_len)
    train_loader = DataLoader(train_seq, batch_size=batch_size, shuffle=True, num_workers=8)

    val_seq = Riiid_Sequence(val_group, seq_len)
    val_loader = DataLoader(val_seq, batch_size=batch_size, shuffle=False, num_workers=8)

    model = SaintPlus(seq_len=seq_len, num_layers=num_layers, d_ffn=d_ffn, d_model=d_model, num_heads=num_heads,
                    max_len=max_len, n_questions=n_questions, n_tasks=n_tasks, dropout=dropout)

    loss_fn = nn.BCELoss()
    if args.optimizer == "adam":
        optimizer = NoamOpt(d_model, 1, 4000 ,optim.Adam(model.parameters(), lr=args.lr))

    model.to(device)
    loss_fn.to(device)

    train_losses = []
    val_losses = []
    val_aucs = []
    best_auc = 0
    count=0
    for e in range(epochs):
        print("==========Epoch {} Start Training==========".format(e+1))
        model.train()
        t_s = time.time()
        train_loss = []
        train_labels = []
        train_preds = []
        for step, data in enumerate(train_loader):
            content_ids = data[0].to(device).long()
            time_lag = data[1].to(device).float()
            ques_elapsed_time = data[2].to(device).float()
            itemaver = data[3].to(device).float()
            useraver = data[4].to(device).float()
            answer_correct = data[5].to(device).long()
            label = data[6].to(device).float()
            optimizer.optimizer.zero_grad()

            preds = model(content_ids, time_lag, ques_elapsed_time, itemaver,useraver, answer_correct)
            loss_mask = (answer_correct != 0)
            preds_masked = torch.masked_select(preds, loss_mask)
            label_masked = torch.masked_select(label, loss_mask)
            loss = loss_fn(preds_masked, label_masked)

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            train_labels.extend(label_masked.view(-1).data.cpu().numpy())
            train_preds.extend(preds_masked.view(-1).data.cpu().numpy())

        train_loss = np.mean(train_loss)
        train_auc = roc_auc_score(train_labels, train_preds)
        
        print("==========Epoch {} Start Validation==========".format(e+1))
        model.eval()
        val_loss = []
        val_labels = []
        val_preds = []

        for step, data in enumerate(val_loader):
            content_ids = data[0].to(device).long()
            time_lag = data[1].to(device).float()
            ques_elapsed_time = data[2].to(device).float()
            itemaver = data[3].to(device).float()
            useraver = data[4].to(device).float()
            answer_correct = data[5].to(device).long()
            label = data[6].to(device).float()

            preds = model(content_ids, time_lag, ques_elapsed_time, itemaver,useraver,answer_correct)
            loss_mask = (answer_correct != 0)
            preds_masked = torch.masked_select(preds, loss_mask)
            label_masked = torch.masked_select(label, loss_mask)

            val_loss.append(loss.item())
            val_labels.extend(label_masked.view(-1).data.cpu().numpy())
            val_preds.extend(preds_masked.view(-1).data.cpu().numpy())

        val_loss = np.mean(val_loss)
        val_auc = roc_auc_score(val_labels, val_preds)
        
        # auc 비교
        if val_auc > best_auc:
            print("Save model at epoch {}".format(e+1))
            save_model(model)
            best_auc = val_auc
            count=0
        else:
            count+=1

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_aucs.append(val_auc)
        exec_t = int((time.time() - t_s)/60)
        print("Train Loss {:.4f}/ Val Loss {:.4f}, Val AUC {:.4f} / Exec time {} min".format(train_loss, val_loss, val_auc, exec_t))
        # patience번 성능 향상이 없다면
        if count==patience:
            print('early stop',best_auc)
            break
    return best_auc

if __name__=="__main__":
    train()