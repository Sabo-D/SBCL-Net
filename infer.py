import torch
from src.dataset import infer_dataloader
from src.models.ResNet101_MFA import PVT_MFA
from src.models.model_ablate_v0 import  PVT_MFA_ablate_v0
from src.models.model_ablate_v1 import  PVT_MFA_ablate_v1
from src.models.model_ablate_v2 import  PVT_MFA_ablate_v2
from src.model_infer import model_infer


if __name__ =='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PVT_MFA(1).to(device)
    model_weight = r"D:\AA_Pycharm_Projs\PVT_MFA\outputs\JS\best_JS.pth"
    model.load_state_dict(torch.load(model_weight, map_location=device))

    image_dir = r"C:\Users\Administrator\Desktop\data\visual_data\data\JS\images_01"
    infer_dataloader = infer_dataloader(image_dir)

    out_path = r"C:\Users\Administrator\Desktop\data\out\feature_map"
    model_infer(model, infer_dataloader, device, out_path)