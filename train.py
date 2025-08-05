import torch
from src.dataset import train_val_dataloader
from src.models.ResNet101_MFA import ResNet_MFA
from src.model_train import model_train
from src.utils.utils import plot_metrics

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet_MFA(1).to(device)

    target = "JS"
    train_image_dir = fr"C:\Users\Administrator\Desktop\data\{target}\train\images"
    train_mask_dir = fr"C:\Users\Administrator\Desktop\data\{target}\train\masks"
    train_edge_dir = fr"C:\Users\Administrator\Desktop\data\{target}\train\edges"
    train_dist_dir = fr"C:\Users\Administrator\Desktop\data\{target}\train\dist_masks"

    valid_image_dir = fr"C:\Users\Administrator\Desktop\data\{target}\valid\images"
    valid_mask_dir = fr"C:\Users\Administrator\Desktop\data\{target}\valid\masks"
    valid_edge_dir = fr"C:\Users\Administrator\Desktop\data\{target}\valid\edges"
    valid_dist_dir = fr"C:\Users\Administrator\Desktop\data\{target}\valid\dist_masks"
    batch_size = 8
    train_dataloader, val_dataloader = train_val_dataloader(train_image_dir, train_mask_dir, train_dist_dir, train_edge_dir,
                                                            valid_image_dir, valid_mask_dir, valid_dist_dir, valid_edge_dir,
                                                            batch_size=batch_size)

    model_path = f"D:\AA_Pycharm_Projs\PVT_MFA\outputs\JS_CNN"
    log_path   = f"D:\AA_Pycharm_Projs\PVT_MFA\outputs\JS_CNN"
    train_precess = model_train(model, train_dataloader, val_dataloader, device,
                                num_epochs=100, model_path=model_path,
                                log_path=log_path)
    plot_metrics(train_precess, 'train', log_path)