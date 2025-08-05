import copy
import sys
import time
from tqdm import tqdm

from src.loss import MultiLoss
from src.utils.utils import *
import os
import pandas as pd
from datetime import datetime
from torch.optim.lr_scheduler import LambdaLR, StepLR


def model_train(model, train_dataloader, val_dataloader, device,
                num_epochs, model_path, log_path):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    warmup_epochs = 5
    def warmup_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        return 1.0

    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)  # warm_up调度器
    step_scheduler = StepLR(optimizer, step_size=30, gamma=0.1)      # 主调度器

    criterion = MultiLoss()

    best_model_wts = copy.deepcopy(model.state_dict())
    train_loss_all, val_loss_all, best_loss_all = [], [], []
    train_iou_all, val_iou_all, best_iou_all = [], [], []
    train_f1_all, val_f1_all = [], []
    train_precision_all, val_precision_all = [], []
    train_recall_all, val_recall_all = [], []
    best_iou = 0.0
    best_loss = 10000000

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        train_loss, val_loss = 0.0, 0.0
        train_iou, val_iou = 0.0, 0.0
        train_f1, val_f1 = 0.0, 0.0
        train_precision, val_precision = 0.0, 0.0
        train_recall, val_recall = 0.0, 0.0
        train_num, val_num = 0, 0

        since = time.time()

        model.train()
        for sample in tqdm(train_dataloader, desc=f'Train Epoch {epoch + 1}', file=sys.stdout):
            images = sample[0].to(device)
            masks = sample[1].to(device)
            dists = sample[2].to(device)
            edges = sample[3].to(device)
            batch_num = images.shape[0]

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs[0] ,outputs[1], masks, edges)
            loss.backward()
            optimizer.step()

            iou,_ = compute_iou_binary(outputs[0], masks)
            metrics = binary_classification_metrics(outputs[0], masks)
            precision = metrics['precision']
            recall = metrics['recall']
            f1 = metrics['f1']

            train_loss += loss.detach().item() * batch_num  # 需要计算epoch的loss 非batch
            train_iou += iou * batch_num
            train_f1 += f1 * batch_num
            train_precision += precision * batch_num
            train_recall += recall * batch_num
            train_num += batch_num

        with torch.no_grad():
            model.eval()
            for sample in tqdm(val_dataloader, desc=f'Valid Epoch {epoch + 1}', file=sys.stdout):
                images = sample[0].to(device)
                masks = sample[1].to(device)
                dists = sample[2].to(device)
                edges = sample[3].to(device)
                batch_num = images.shape[0]

                outputs = model(images)
                loss = criterion(outputs[0], outputs[1], masks, edges)

                iou,_ = compute_iou_binary(outputs[0], masks)
                metrics = binary_classification_metrics(outputs[0], masks)
                precision = metrics['precision']
                recall = metrics['recall']
                f1 = metrics['f1']

                val_iou += iou * batch_num
                val_loss += loss.detach().item() * batch_num
                val_f1 += f1 * batch_num
                val_precision += precision * batch_num
                val_recall += recall * batch_num
                val_num += batch_num

        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            step_scheduler.step()

        train_loss_all.append(train_loss / train_num)
        val_loss_all.append(val_loss / val_num)
        train_iou_all.append(train_iou / train_num)
        val_iou_all.append(val_iou / val_num)
        train_f1_all.append(train_f1 / train_num)
        val_f1_all.append(val_f1 / val_num)
        train_precision_all.append(train_precision / train_num)
        val_precision_all.append(val_precision / val_num)
        train_recall_all.append(train_recall / train_num)
        val_recall_all.append(val_recall / val_num)

        if val_loss / val_num < best_loss:
            best_loss = val_loss / val_num
            best_model_wts_loss = copy.deepcopy(model.state_dict())
            model_name = f"best_loss_epoch_{epoch + 1}.pth"
            model_save_path = os.path.join(model_path, model_name)
            torch.save(best_model_wts_loss, model_save_path)
        if val_iou / val_num > best_iou:
            best_iou = val_iou / val_num
            best_model_wts_iou = copy.deepcopy(model.state_dict())
            model_name = f"best_iou_epoch_{epoch + 1}.pth"
            model_save_path = os.path.join(model_path, model_name)
            torch.save(best_model_wts_iou, model_save_path)
        best_iou_all.append(best_iou)
        best_loss_all.append(best_loss)


        time_elapsed = time.time() - since
        print('Train loss:{:.4f}  Train iou:{:.4f}  Train f1:{:.4f}  Train precision:{:.4f}  Train recall:{:.4f}'\
              .format(train_loss_all[-1], train_iou_all[-1], train_f1_all[-1], train_precision_all[-1], train_recall_all[-1]))
        print('Valid loss:{:.4f}  Valid iou:{:.4f}  Valid f1:{:.4f}  Valid precision:{:.4f}  Valid recall:{:.4f}' \
              .format(val_loss_all[-1], val_iou_all[-1], val_f1_all[-1], val_precision_all[-1], val_recall_all[-1]))
        print('Training time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    train_process = pd.DataFrame(data={
        "epoch": range(1, num_epochs + 1),
        "train_loss": train_loss_all,
        "train_iou": train_iou_all,
        "train_f1": train_f1_all,
        "train_precision": train_precision_all,
        "train_recall": train_recall_all,
        "val_loss": val_loss_all,
        "val_iou": val_iou_all,
        "val_f1": val_f1_all,
        "val_precision": val_precision_all,
        "val_recall": val_recall_all,
        "best_loss":best_loss_all,
        "best_iou": best_iou_all,
    })
    train_process = train_process.round(4)
    cur_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    log_name =f"log_{cur_time}.csv"
    log_save_path = os.path.join(log_path, log_name)
    train_process.to_csv(log_save_path)
    print("Finished Training")

    return train_process







