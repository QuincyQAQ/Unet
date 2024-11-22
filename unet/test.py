import warnings
import os
from torchmetrics.functional.regression import mean_absolute_error
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from torchvision.utils import save_image
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm

from config import Config
from data import get_validation_data
from models import *
from utils import *

warnings.filterwarnings('ignore')

opt = Config('config.yml')

seed_everything(opt.OPTIM.SEED)

os.makedirs('result', exist_ok=True)
criterion_lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).cuda()
def test():
    accelerator = Accelerator()

    # Data Loader
    val_dir = opt.TRAINING.VAL_DIR

    val_dataset = get_validation_data(val_dir, opt.MODEL.INPUT, opt.MODEL.TARGET, {'w': opt.TRAINING.PS_W, 'h': opt.TRAINING.PS_H, 'ori': False})
    testloader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False,
                            pin_memory=True)

    # Model & Metrics
    model = Model()
    print("Current working directory:", os.getcwd())
    weight_dir = "runs/exp7/_epoch_1.pth"
    load_checkpoint(model, weight_dir)

    model, testloader = accelerator.prepare(model, testloader)

    model.eval()

    size = len(testloader)
    psnr = 0
    ssim = 0
    lpips = 0
    mae = 0

    for _, test_data in enumerate(tqdm(testloader, disable=not accelerator.is_local_main_process)):
        tar = (test_data[1] * 255)
        inp = test_data[0].contiguous()
        res = (model(inp)[0].clamp(0, 1) * 255)
        res = res.unsqueeze(0)
        tar = torch.floor(tar)
        inp = torch.floor(inp)
        res = torch.floor(res)


        psnr += peak_signal_noise_ratio(res, tar, data_range=255)
        ssim += structural_similarity_index_measure(res, tar, data_range=255)

        tar = test_data[1]
        inp = test_data[0].contiguous()
        res = model(inp)[0].clamp(0, 1)
        res = res.unsqueeze(0)

        mae += mean_absolute_error(torch.mul(res, 255), torch.mul(tar, 255))
        lpips += criterion_lpips(res, tar).item()

    # print(f"testloader: {len(testloader)}")
    psnr /= len(testloader)
    ssim /= len(testloader)
    mae /= len(testloader)
    lpips /= len(testloader)


    print("PSNR: {}, SSIM: {}, MAE: {}, LPIPS: {}".format(psnr, ssim, mae, lpips))


if __name__ == '__main__':
    test()
