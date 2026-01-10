# main program 
# 
# run_SLM.py
# run_SLM.slurm

import os, os.path as osp
import glob 
import shutil 
import random
import logging
import datetime
import numpy as np
import torch
import argparse
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange
import cv2
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from torch.fft import fft2, ifftshift, ifft2, fftshift
from diffuser import DiffuserProvider
from dataset import MNISTDataset
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--expid', default='dyna_z_pcc')
parser.add_argument('--device', default='cuda:0', type=str)
parser.add_argument('--bs', default=16, type=int)
parser.add_argument('--lr', default=0.00099, type=float)
parser.add_argument('--gamma', default=0.99, type=float)
parser.add_argument('--disable_batch_mode', action='store_true')
parser.add_argument('--num_epochs', default=50, type=int)
parser.add_argument('--num_max_objects', default=-1, type=int)
parser.add_argument('--diffusers_per_epoch', default=10, type=int)
parser.add_argument('--test_diffusers_per_epoch', default=10, type=int)    
parser.add_argument('--z1_multiplier', default=0.02, type=float)
parser.add_argument('--load_phase_plane', default='', type=str)
parser.add_argument('--eval', action='store_true')

args = parser.parse_args()

# Network training parameters
num_epochs = args.num_epochs
batch_size = args.bs
lr = args.lr
gamma = args.gamma
device = args.device
show_interval = 700
evalonly = args.eval
load_phase_plane = args.load_phase_plane
z1_multiplier = args.z1_multiplier
num_max_objects = args.num_max_objects
diffusers_per_epoch = args.diffusers_per_epoch
test_diffusers_per_epoch = args.test_diffusers_per_epoch     
# Simulation results save path
savedir = osp.join("results", "ours", "2025_6_3", "elight_new_scatter_2layer_SLM_256_780nm_diff_after","2kernel", args.expid)  
dyna_diffuser = True

for subdir in ['train', 'test', 'phase_plane']:
    os.makedirs(osp.join(savedir, subdir), exist_ok=True)
enable_batch_mode = not args.disable_batch_mode

LOGFORMAT = "%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s"
nowtime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
logging.basicConfig(filename=osp.join(savedir, "{}.log".format(nowtime)),
                    level=logging.INFO, format=LOGFORMAT)
stderrLogger = logging.StreamHandler()
stderrLogger.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
logging.getLogger().addHandler(stderrLogger)
logging.info('to be saved at {}'.format(savedir))
logging.info(vars(args))
# TensorBoard log directory path
tbdir = osp.join("runs", "ours",  "2025_6_3", "elight_new_scatter_2layer_SLM_256_780nm_diff_after", "2kernel", args.expid + '_' + nowtime)
os.makedirs(tbdir, exist_ok=True)
summary_writer = SummaryWriter(logdir=tbdir)

diffuser_rng = random.Random(2025)  # Change the number to alter the scattering media loading order.
random.seed(2023)                   # Change the number to alter the objects loading order.

# Change the number to modify the network initialization parameters.
np.random.seed(2024)
torch.manual_seed(2024)
torch.cuda.manual_seed_all(2024)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Function definition
#================================================================================================
#================================================================================================
def ift2(G, delta_f, batch_mode=False):
    N = G.shape[1] if batch_mode else G.shape[0]
    g = ifftshift(ifft2(fftshift(G))) * ((N * delta_f)**2)
    return g

def ft2(g, delta):
    G = ifftshift(fft2(ifftshift(g))) * (delta**2)
    return G

# Free-space propagation
def ang_spec_prop_batch(Uin, wvl, d1, d2, Dz, device="cuda:0"):
    bs, N = Uin.shape[:2]
    k = 2 * torch.pi / wvl
    xs, ys = torch.arange(-N / 2, N / 2).to(device), torch.arange(-N / 2, N / 2).to(device)
    #  Source plane coordinates
    x1, y1 = torch.meshgrid(xs * d1, ys * d1)
    r1sq = x1 ** 2 + y1 ** 2
    r1sq = r1sq[None].repeat((bs, 1, 1))
    # Spatial frequencies in the source plane
    df1 = 1 / (N * d1)
    fX, fY = torch.meshgrid(xs * df1, ys * df1)
    fsq = fX ** 2 + fY ** 2
    fsq = fsq[None].repeat((bs, 1, 1))
    # Scaling parameters
    m = d2 / d1
    # Observation plane coordinates
    x2, y2 = torch.meshgrid(xs * d2, ys * d2)
    # r2sq \in [H, W] => [1, H, W] => [B, H, W]
    # r2sq[0] and r2sq[1] are the same
    r2sq = x2 ** 2 + y2 ** 2
    r2sq = r2sq[None].repeat((bs, 1, 1))
    # Quadratic phase factor
    Q1 = torch.exp(1j * k / 2 * (1 - m) / Dz * r1sq).to(device)
    Q2 = torch.exp(-1j * torch.pi ** 2 * 2 * Dz / m / k * fsq).to(device)
    Q3 = torch.exp(1j * k / 2 * (m - 1) / (m * Dz) * r2sq).to(device)
    # Calculate the propagated optical field
    # import pdb; pdb.set_trace()
    Uout = Q3 * ift2(Q2 * ft2(Q1 * Uin / m, d1), df1, batch_mode=True)
    return x2, y2, Uout



# forward propagation process in optical diffraction neural networks
def forward_func_batch(obj, phase_plane1_exp, phase_plane2_exp, diffuser, mode='ref',
                 device="cuda:0", prop_func=ang_spec_prop_batch):
    # TODO: obj [H x W]
    # multiplier0 = ap_diff[None] * diffuser if mode != 'ref' else Trans_plane[None]
    multiplier0 = diffuser if mode != 'ref' else Trans_plane[None]
    multiplier1 = torch.exp(1j * (phase_plane1_exp[None] % (2 * torch.pi))) if mode == 'modulated' else Trans_plane[None]
    multiplier2 = torch.exp(1j * (phase_plane2_exp[None] % (2 * torch.pi))) if mode == 'modulated' else Trans_plane[None]

    _, _, Uout1 = prop_func(obj, lmd, d1, d2, z1, device=device)
    Uout2 = Uout1 * multiplier1
    _, _, Uout3 = prop_func(Uout2, lmd, d2, d3, z2, device=device)
    Uout4 = Uout3 * multiplier2
    _, _, Uout5 = prop_func(Uout4, lmd, d3, d4, z3, device=device)
    Uout6 = Uout5 * multiplier0
    _, _, Uout7 = prop_func(Uout6, lmd, d4, d5, z4, device=device)
    
    I = torch.abs(Uout7) ** 2
    return I

def calculate_loss(I_ref, I_pred, alpha=1, beta=1.0):
    assert I_ref.ndim == 3
    bs, h, w = I_ref.shape[:3]
    I_ref_normalized = I_ref / (I_ref.max() + 1e-8)
    I_pred_normalized = I_pred / (I_pred.max() + 1e-8)
    # import ipdb; ipdb.set_trace()
    # Energy penalty term
    energy_penalty = (alpha * (1 - I_ref_normalized) * I_pred_normalized - beta * I_ref_normalized * I_pred_normalized).sum() / (I_ref_normalized.sum())
    energy_penalty /= bs

    # PCC term
    I_pred = I_pred / I_pred.max()
    I_ref_mean = I_ref.mean(dim=[1, 2], keepdim=True)
    I_pred_mean = I_pred.mean(dim=[1, 2], keepdim=True)
    pcc_loss = ((I_ref - I_ref_mean) * (I_pred - I_pred_mean)).sum(dim=[1, 2]) / torch.sqrt((I_ref - I_ref_mean).pow(2).sum(dim=[1, 2]) * (I_pred - I_pred_mean).pow(2).sum(dim=[1, 2]))
    pcc_loss = -pcc_loss.sum() / bs
    # import ipdb; ipdb.set_trace()
    return pcc_loss, energy_penalty

#================================================================================================
#================================================================================================
# Simulation parameter settings

# Wavelength
lmd = 780 * 1e-9
# Pixel size
d1 = 12.5 * 2 * 1e-6;    d2 = 12.5 * 2 * 1e-6;   d3 = 12.5 * 2 * 1e-6;   d4 = 12.5 * 2 * 1e-6;    d5 = 12.5 * 2 * 1e-6
# Propagation distance between adjacent planes
z1 = z1_multiplier;      z2 = 0.05 * 4;          z3 = 0.05 * 4;       z4 = 0.05 * 4
# Number of pixels
N = 256
# Plane size
D1 = N * d1
D2 = N * d2
D3 = N * d3
D4 = N * d4
D5 = N * d5


fresnel_number = ((D1 / 2) ** 2) / (lmd * z1)
logging.info("fresnel_number: {}".format(fresnel_number))


Trans_plane = torch.ones((N, N)).to(device)


# Load the training dataset (objects and scattering media)
train_object_folder, train_diffuser_folder = "mnist_data/mnist_256/mnist_complete_train_166_256_2w_tr.pkl", "diffuser_data/diff_elight/diff_256/train_elight_new_diffuser_256_2kernel"
train_dataset = MNISTDataset(train_object_folder, num_max_objects=num_max_objects, target_size=400)
train_diffuser_provider = DiffuserProvider(train_diffuser_folder, diffusers_per_epoch=diffusers_per_epoch, rng=diffuser_rng)

# Load the test dataset (objects and scattering media)
test_object_folder, test_diffuser_folder = "mnist_data/mnist_256/mnist_complete_test_166_256_tr.pkl", "diffuser_data/diff_elight/diff_256/test_elight_new_diffuser_256_2kernel"
test_dataset = MNISTDataset(test_object_folder, num_max_objects=num_max_objects, target_size=400)
test_diffuser_provider = DiffuserProvider(test_diffuser_folder, diffusers_per_epoch=test_diffusers_per_epoch, rng=diffuser_rng)    

logging.info("lambda: {}, N: {}".format(lmd, N))
logging.info("train object folder: {}, test object folder: {}".format(train_object_folder, test_object_folder))
logging.info("train diffuser folder: {}, test diffuser folder: {}".format(train_diffuser_folder, test_diffuser_folder))

for name, value in zip(["z1", "z2", "z3", "z4"], [z1, z2, z3, z4]):
    logging.info("{} : {} ({:.4f})".format(name, value, value / lmd))

for name, value in zip(['d1', 'd2', 'd3', 'd4', 'd5'], [d1, d2, d3, d4, d5]):
    logging.info("{} : {}".format(name, value))

for name, value in zip(['D1', 'D2', 'D3', 'D4', 'D5'], [D1, D2, D3, D4, D5]):
    logging.info("{} : {}".format(name, value))

# backup the file 
script_backup_path = osp.join(savedir, "{}_{}.py".format(osp.basename(__file__).split('.')[0], nowtime))
if not osp.exists(script_backup_path):
    shutil.copy(os.path.abspath(__file__), script_backup_path)

# Diffraction planes (two layers, as an example)
phase_plane1_exp = torch.ones(N, N).to(device).requires_grad_()
phase_plane2_exp = torch.ones(N, N).to(device).requires_grad_()

if load_phase_plane:
    phase_plane_paths = glob.glob(osp.join(load_phase_plane, "*.npy"))
    phase_plane_paths = sorted(phase_plane_paths, key=lambda x: int(osp.basename(x).split('_')[0]))[-4:]
    phase_plane_paths = sorted(phase_plane_paths, key=lambda x: int(osp.basename(x).split('.')[0].split('_')[1]))
    logging.info("phase plane to be loaded : {}".format(phase_plane_paths))
    phase_planes = [np.load(phase_plane_path) for phase_plane_path in phase_plane_paths]
    phase_plane1_exp = torch.exp(1j * torch.from_numpy(phase_planes[0]).float()).to(device)
    phase_plane2_exp = torch.exp(1j * torch.from_numpy(phase_planes[1]).float()).to(device)
    

optimizer = torch.optim.Adam([phase_plane1_exp, phase_plane2_exp], lr=lr)
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

total_iters = 0
for epoch in range(num_epochs):
    # define dataloaders
    if not evalonly:
        train_diffuser_image = train_diffuser_provider.get_packed_diffusers().to(device)
        num_train_diffusers = train_diffuser_image.shape[0]
        train_diffuser_image = train_diffuser_image.repeat(batch_size, 1, 1)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        tbar = tqdm(train_loader, dynamic_ncols=True)
        losses = []
        train_cnt = 0
        for train_object_image in tbar:
            optimizer.zero_grad()
            rbs, rh, rw = train_object_image.shape[:3]
            train_object_image = train_object_image.to(device)[:, None].repeat(1, num_train_diffusers, 1, 1).view(-1, rh, rw)
            #  Result after scattering and ODNN correction
            I_modulated = forward_func_batch(train_object_image, phase_plane1_exp, phase_plane2_exp, train_diffuser_image, mode='modulated', device=device, prop_func=ang_spec_prop_batch)


            if train_cnt % show_interval == 0:
                # Result after scattering without ODNN correction
                I_unmodulated_but_diffused = forward_func_batch(train_object_image, phase_plane1_exp, phase_plane2_exp,
                                                                train_diffuser_image, mode='unmodulated', device=device,
                                                                prop_func=ang_spec_prop_batch)
                # Ground truth (objects)
                img0 = np.concatenate([iimg.cpu().numpy() for iimg in train_object_image], axis=1); img0 = (img0 - img0.min()) / (img0.max() - img0.min())

                img2 = I_modulated.permute(0, 2, 1).contiguous().view(-1, I_modulated.shape[-1]).permute(1, 0).contiguous();
                img2 = (img2 - img2.min()) / (img2.max() - img2.min())
                img3 = I_unmodulated_but_diffused.permute(0, 2, 1).contiguous().view(-1, I_unmodulated_but_diffused.shape[
                    -1]).permute(1, 0).contiguous();
                img3 = (img3 - img3.min()) / (img3.max() - img3.min())
                I_vis = np.concatenate((img0,
                                        img2.detach().cpu().numpy(),
                                        img3.cpu().numpy()), axis=0)
                savepath = osp.join(savedir, "train", "{}_{}.png".format(epoch, train_cnt))
                lh, lw = I_vis.shape[:2]
                I_vis_small = cv2.resize(I_vis, (lw // 8, lh // 8))
                plt.imsave(savepath, I_vis, cmap='gray')
                # plt.imsave(savepath.replace(".png", "_small.png"), I_vis_small, cmap='gray')
            
            # object rotate 90 degree
            ####################----rotate---################################################
            # train_object_image = train_object_image.flip(1).flip(2).contiguous()
            # train_object_image = train_object_image.permute(0, 2, 1).flip(2).contiguous()  
            #################################################################################
            # Object scaling
            #########################----scale----#####################################################################################
            # train_object_image = train_object_image.unsqueeze(1)  
            # train_object_image_interp = F.interpolate(train_object_image, size=(128, 128), mode='bilinear', align_corners=False) 
            # train_object_image_padded = F.pad(train_object_image_interp, (64, 64, 64, 64), mode='constant', value=0) 
            # train_object_image = train_object_image_padded.squeeze(1) # 
            ###########################################################################################################################
            pccloss, energy = calculate_loss(train_object_image, I_modulated)
            # loss = pccloss + energy
            loss = pccloss
            summary_writer.add_scalar('pcc_loss/iters', pccloss.item(), total_iters)
            summary_writer.add_scalar('energy_penalty/iters', energy.item(), total_iters)
            summary_writer.add_scalar('loss/iters', loss.item(), total_iters)

            loss.backward()
            tbar.set_postfix(loss=loss.item(), pccloss=pccloss.item(), energy=energy.item())
            losses.append(loss.item())
            optimizer.step()
            train_cnt += 1
            total_iters += 1

        scheduler.step()
        if dyna_diffuser:
            train_diffuser_provider.switch_diffusers()

    phase_planes_tosave = [phase_plane1_exp, phase_plane2_exp] 
    for phaseid, phase_plane in enumerate(phase_planes_tosave):
        savepath = osp.join(savedir, "phase_plane", str(epoch) + '_' + str(phaseid + 1) + '.npy')
        phase = phase_plane.detach().cpu().numpy() % (2 * np.pi)
        np.save(savepath, phase)
        savemat(savepath.replace(".npy", ".mat"), {"data": phase})
        plt.figure()
        plt.imshow(phase)
        plt.axis('off')
        plt.colorbar()
        plt.savefig(savepath.replace(".npy", ".png"), dpi=220)
        plt.close()

    test_diffuser_image = test_diffuser_provider.get_packed_diffusers().to(device)
    num_test_diffusers = test_diffuser_image.shape[0]
    test_diffuser_image = test_diffuser_image.repeat(batch_size, 1, 1)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # evaluation
    with torch.no_grad():
        cnt = 0
        total_eval_loss = 0.0
        epoch_losses = [] # Record all loss values during an epoch.
        tbar = tqdm(test_loader, total=len(test_loader), dynamic_ncols=True)
        for test_object_image in tbar:
            rbs, rh, rw = test_object_image.shape[:3]
            test_object_image = test_object_image.to(device)[:, None].repeat(1, num_test_diffusers, 1, 1).view(-1, rh, rw)

            I_modulated = forward_func_batch(test_object_image, phase_plane1_exp, phase_plane2_exp, test_diffuser_image, mode='modulated', device=device, prop_func=ang_spec_prop_batch)

            if batch_size == 1 or cnt % show_interval == 0:
                I_unmodulated_but_diffused = forward_func_batch(test_object_image, phase_plane1_exp, phase_plane2_exp, test_diffuser_image, mode='unmodulated', device=device, prop_func=ang_spec_prop_batch)
  
                img0 = np.concatenate([iimg.cpu().numpy() for iimg in test_object_image], axis=1); img0 = (img0 - img0.min()) / (img0.max() - img0.min())
                # img1 = I_ref.permute(0, 2, 1).contiguous().view(-1, I_ref.shape[-1]).permute(1, 0).contiguous();
                # img1 = (img1 - img1.min()) / (img1.max() - img1.min())
                img2 = I_modulated.permute(0, 2, 1).contiguous().view(-1, I_modulated.shape[-1]).permute(1, 0).contiguous();
                img2 = (img2 - img2.min()) / (img2.max() - img2.min())
                img3 = I_unmodulated_but_diffused.permute(0, 2, 1).contiguous().view(-1, I_unmodulated_but_diffused.shape[
                    -1]).permute(1, 0).contiguous();
                img3 = (img3 - img3.min()) / (img3.max() - img3.min())

                I_vis = np.concatenate((img0, img2.detach().cpu().numpy(),
                                        img3.cpu().numpy()), axis=0)
                lh, lw = I_vis.shape[:2]
                I_vis_small = cv2.resize(I_vis, (lw // 8, lh // 8))

                plt.imsave(osp.join(savedir, "test", "{}_{}.png".format(epoch, cnt)), I_vis, cmap='gray')
                # plt.imsave(osp.join(savedir, "test", "{}_{}_small.png".format(epoch, cnt)), I_vis_small, cmap='gray')

            #####################----rotate---#################################################
            # test_object_image = test_object_image.flip(1).flip(2).contiguous()
            # test_object_image = test_object_image.permute(0, 2, 1).flip(2).contiguous()
            ###################################################################################
            
            #########################----scale----#####################################################################################
            # test_object_image = test_object_image.unsqueeze(1)  
            # test_object_image_interp = F.interpolate(test_object_image, size=(128, 128), mode='bilinear', align_corners=False) 
            # test_object_image_padded = F.pad(test_object_image_interp, (64, 64, 64, 64), mode='constant', value=0) 
            # test_object_image = test_object_image_padded.squeeze(1) # 
            ###########################################################################################################################
            pccloss, energy = calculate_loss(test_object_image, I_modulated)
            # loss = pccloss + energy
            loss = pccloss
            cnt += 1
            total_eval_loss += loss.item()
            epoch_losses.append(loss.item())

        total_eval_loss /= len(test_loader)
        epoch_losses = np.array(epoch_losses)
        loss_variance = np.var(epoch_losses)
        logging.info("evaluation of epoch {} finished, total loss: {:.6f} variance: {:.8f}".format(epoch, total_eval_loss, loss_variance))
        summary_writer.add_scalar('eval_loss/epoch', total_eval_loss, epoch)


    # plt.imsave(osp.join(savedir, "test", "{}_{}.png".format(epoch, cnt)), I_vis, cmap='gray')
    if evalonly:
        break





