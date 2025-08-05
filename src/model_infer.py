import sys
import time
from tqdm import tqdm
from src.utils.utils import *
from src.visual.visual_featuremap import visualize_feature_response

def model_infer(model, infer_dataloader, device, out_path):
    model.to(device)
    since = time.time()

    with torch.no_grad():
        model.eval()
        for sample in tqdm(infer_dataloader, desc='Infer', file=sys.stdout):
            # sample : image, name, geo_trans, geo_proj
            image = sample[0].to(device)
            name = sample[1][0]

            outputs = model(image)
            image_out = outputs[0]
            f_edge = outputs[2]
            visualize_feature_response(f_edge, mode='average', save_path=name+'f_beb_after_ave.png',cmap="jet")
            visualize_feature_response(f_edge, mode='max', save_path=name+'f_beb_after_max.png',cmap="jet")
            mask_ndarray = out_to_ndarray(image_out)
            out_save_path = os.path.join(out_path, name+'.png')
            save_with_geo(out_save_path, sample[2], sample[3][0], mask_ndarray)

    time_elapsed = time.time() - since
    print('Testing time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Finished infer')

    return 0


