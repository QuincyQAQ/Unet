import warnings

import torch.optim as optim
from accelerate import Accelerator
from pytorch_msssim import SSIM
from torch.utils.data import DataLoader
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from tqdm import tqdm
from torchmetrics.functional.regression import mean_absolute_error
from config import Config
from data import get_training_data, get_validation_data
from models import *
from utils import *
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import csv
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

opt = Config('config.yml')

seed_everything(opt.OPTIM.SEED)

def get_next_exp_folder(base_dir="runs/exp"):
    exp_num = 1
    while os.path.exists(f"{base_dir}{exp_num}"):
        exp_num += 1
    exp_path = f"{base_dir}{exp_num}"
    print(f"Creating experiment folder at: {exp_path}")  # Debugging output
    os.makedirs(exp_path, exist_ok=True)
    return exp_path


def plot_metrics(csv_file,exp_folder):
    # Read the metrics from the CSV file
    epochs = []
    train_loss = []
    psnr = []
    ssim = []
    mae = []
    lpips = []
    best_psnr = []

    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            epochs.append(int(row[0]))  # Epoch
            train_loss.append(float(row[1]))  # Train loss
            psnr.append(float(row[2]))  # PSNR
            ssim.append(float(row[3]))  # SSIM
            mae.append(float(row[4]))  # MAE
            lpips.append(float(row[5]))  # LPIPS
            best_psnr.append(float(row[6]))  # Best PSNR

    # Plotting the results
    plt.figure(figsize=(10, 8))

    # Plot PSNR
    plt.subplot(2, 2, 1)
    plt.plot(epochs, psnr, label='PSNR', color='tab:blue')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR')
    plt.title('PSNR vs Epoch')
    plt.grid(True)

    # Plot SSIM
    plt.subplot(2, 2, 2)
    plt.plot(epochs, ssim, label='SSIM', color='tab:orange')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.title('SSIM vs Epoch')
    plt.grid(True)

    # Plot MAE
    plt.subplot(2, 2, 3)
    plt.plot(epochs, mae, label='MAE', color='tab:green')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('MAE vs Epoch')
    plt.grid(True)

    # Plot LPIPS
    plt.subplot(2, 2, 4)
    plt.plot(epochs, lpips, label='LPIPS', color='tab:red')
    plt.xlabel('Epoch')
    plt.ylabel('LPIPS')
    plt.title('LPIPS vs Epoch')
    plt.grid(True)

    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(exp_folder, "metrics_plot.png"))
    plt.show()

def train():
    # Accelerate

    exp_folder = get_next_exp_folder()
    metrics_file = os.path.join(exp_folder, "metrics.csv") if exp_folder else "metrics.csv"

    with open(metrics_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["epoch", "train_loss", "PSNR", "SSIM", "MAE", "LPIPS", "best_PSNR"])

    accelerator = Accelerator(log_with='wandb') if opt.OPTIM.WANDB else Accelerator()
    device = accelerator.device
    config = {
        "dataset": opt.TRAINING.TRAIN_DIR,
        "model": opt.MODEL.SESSION
    }
    accelerator.init_trackers("shadow", config=config)

    if accelerator.is_local_main_process:
        os.makedirs(opt.TRAINING.SAVE_DIR, exist_ok=True)

    # Data Loader
    train_dir = opt.TRAINING.TRAIN_DIR
    val_dir = opt.TRAINING.VAL_DIR

    train_dataset = get_training_data(train_dir, opt.MODEL.INPUT, opt.MODEL.TARGET, {'w': opt.TRAINING.PS_W, 'h': opt.TRAINING.PS_H})
    trainloader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, num_workers=16,
                             drop_last=False, pin_memory=True)
    val_dataset = get_validation_data(val_dir, opt.MODEL.INPUT, opt.MODEL.TARGET, {'w': opt.TRAINING.PS_W, 'h': opt.TRAINING.PS_H, 'ori': opt.TRAINING.ORI})
    testloader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False,
                            pin_memory=True)

    # Model & Loss
    model = Model()
    criterion_ssim = SSIM(data_range=1, size_average=True, channel=3).to(device)
    criterion_psnr = torch.nn.MSELoss()
    criterion_lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).cuda()

    # Optimizer & Scheduler
    optimizer_b = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.OPTIM.LR_INITIAL, betas=(0.9, 0.999), eps=1e-8)
    scheduler_b = optim.lr_scheduler.CosineAnnealingLR(optimizer_b, opt.OPTIM.NUM_EPOCHS, eta_min=opt.OPTIM.LR_MIN)

    trainloader, testloader = accelerator.prepare(trainloader, testloader)
    model = accelerator.prepare(model)
    optimizer_b, scheduler_b = accelerator.prepare(optimizer_b, scheduler_b)


    start_epoch = 1
    best_epoch = 1
    best_psnr = 0
    size = len(testloader)
    # training
    for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
        model.train()

        for i, data in enumerate(tqdm(trainloader, disable=not accelerator.is_local_main_process)):
            # get the inputs; data is a list of [target, input, filename]
            inp = data[0].contiguous()
            tar = data[1]

            # forward
            optimizer_b.zero_grad()
            res = model(inp)

            loss_psnr = criterion_psnr(res, tar)
            loss_ssim = 1 - criterion_ssim(res, tar)

            train_loss = loss_psnr + 0.4 * loss_ssim

            # backward
            accelerator.backward(train_loss)
            optimizer_b.step()

        scheduler_b.step()

        # testing
        if epoch % opt.TRAINING.VAL_AFTER_EVERY == 0:
            model.eval()
            with torch.no_grad():
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

                if psnr > best_psnr:
                    best_psnr = psnr
                    if accelerator.is_local_main_process:
                        save_checkpoint({
                            'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer_b.state_dict(),
                        }, epoch, exp_folder)


                if accelerator.is_local_main_process:
                    with open(metrics_file, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(
                            [epoch, train_loss.item(), psnr.item(), ssim.item(), mae.item(), lpips, best_psnr.item()])
                    gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "4")  # 默认为 0
                    print(
                        f"Epoch: {epoch}, Loss:{train_loss}, PSNR: {psnr:.4f}, SSIM: {ssim:.4f}, MAE: {mae:.4f}, LPIPS: {lpips:.4f}, Best PSNR: {best_psnr:.4f}, GPU ID: {gpu_ids}")

    plot_metrics(metrics_file, exp_folder)


if __name__ == '__main__':
    train()
