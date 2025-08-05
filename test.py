import torch
from src.dataset import test_dataloader
from src.models.ResNet101_MFA import  ResNet_MFA
from src.models.model_ablate_v0 import PVT_MFA_ablate_v0
from src.models.model_ablate_v1 import PVT_MFA_ablate_v1
from src.models.model_ablate_v2 import PVT_MFA_ablate_v2
from src.model_test import model_test


if __name__ =='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet_MFA(1).to(device)
    model_weight = r"D:\AA_Pycharm_Projs\PVT_MFA\outputs\JS_CNN\best_iou_epoch_36.pth"
    model.load_state_dict(torch.load(model_weight, map_location=device))

    target = "JS"
    test_image_dir = f"C:\\Users\Administrator\Desktop\data\\{target}\\test\images"
    test_mask_dir = f"C:\\Users\Administrator\Desktop\data\\{target}\\test\masks"
    test_edge_dir = f"C:\\Users\Administrator\Desktop\data\\{target}\\test\edges"
    test_dist_dir = f"C:\\Users\Administrator\Desktop\data\\{target}\\test\dist_masks"
    test_dataloader = test_dataloader(test_image_dir, test_mask_dir, test_dist_dir, test_edge_dir)

    log_path = rf"D:\AA_Pycharm_Projs\PVT_MFA\outputs\JS_CNN"
    model_test(model, test_dataloader, device ,log_path)