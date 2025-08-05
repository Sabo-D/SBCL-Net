import sys
import time
from tqdm import tqdm

from src.loss import MultiLoss
from src.utils.utils import *
import os
import pandas as pd
from datetime import datetime



def model_test(model, test_dataloader, device, log_path):
    model.to(device)
    criterion = MultiLoss()

    loss_total, iou_total, f1_total, oa_total = 0.0, 0.0, 0.0, 0.0
    precision_total, recall_total, test_num = 0.0, 0.0, 0
    comp_total, corr_total, fbdy_total = 0.0, 0.0, 0.0
    goc_total, guc_total, gtc_total = 0.0, 0.0, 0.0

    since = time.time()

    with torch.no_grad():
        model.eval()
        for sample in tqdm(test_dataloader, desc=f'Testing', file=sys.stdout):
            images = sample[0].to(device)
            masks = sample[1].to(device)
            dists = sample[2].to(device)
            edges = sample[3].to(device)
            test_num += 1

            outputs = model(images)
            loss = criterion(outputs[0] ,outputs[1], masks, edges)

            iou,_ = compute_iou_binary(outputs[0], masks)
            metrics = binary_classification_metrics(outputs[0], masks)
            precision = metrics['precision']
            recall = metrics['recall']
            f1 = metrics['f1']
            oa = metrics['oa']

            pred_binary = preprocess_pred(outputs[0].squeeze())
            mask_binary = preprocess_mask(masks.squeeze())

            fbdy_result = calculate_boundary_metrics(pred_binary, mask_binary)
            goe_result = compute_object_metrics(pred_binary, mask_binary)
            comp = fbdy_result['Completeness']
            corr = fbdy_result['Correctness']
            fbdy = fbdy_result['Fbdy']
            goc = goe_result['GOC']
            guc = goe_result['GUC']
            gtc = goe_result['GTC']

            loss_total += loss
            iou_total += iou
            f1_total += f1
            precision_total += precision
            recall_total += recall
            oa_total += oa

            comp_total += comp
            corr_total += corr
            fbdy_total += fbdy
            goc_total += goc
            guc_total += guc
            gtc_total += gtc

    time_elapsed = time.time() - since
    test_loss = loss_total / test_num
    test_loss = test_loss.cpu().detach().numpy()
    test_iou = iou_total / test_num
    test_f1 = f1_total / test_num
    test_precision = precision_total / test_num
    test_recall = recall_total / test_num
    test_oa = oa_total / test_num

    test_comp = comp_total / test_num
    test_corr = corr_total / test_num
    test_fbdy = fbdy_total / test_num
    test_goc = goc_total / test_num
    test_guc = guc_total / test_num
    test_gtc = gtc_total / test_num

    print('loss:{:.4f}  iou:{:.4f}  f1:{:.4f}  precision:{:.4f}  recall:{:.4f}  oa:{:.4f}'\
          .format(test_loss, test_iou, test_f1, test_precision, test_recall, test_oa, ))
    print('comp:{:.4f}  corr:{:.4f}  fbdy:{:.4f}  goc:{:.4f}  guc:{:.4f}  gtc:{:.4f}' \
          .format(test_comp, test_corr, test_fbdy, test_goc, test_guc, test_gtc, ))
    print('Testing time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    test_process = pd.DataFrame(data={
        "loss": [test_loss],
        "iou": [test_iou],
        "f1": [test_f1],
        "precision": [test_precision],
        "recall": [test_recall],
        'oa': [test_oa],
        'comp': [test_comp],
        'corr': [test_corr],
        'fbdy': [test_fbdy],
        'goc': [test_goc],
        'guc': [test_guc],
        'gtc': [test_gtc],
    })
    test_process = test_process.round(4)
    cur_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    log_name =f"log_{cur_time}.csv"
    log_save_path = os.path.join(log_path, log_name)
    test_process.to_csv(log_save_path)
    print("Finished Testing")

    return 0







