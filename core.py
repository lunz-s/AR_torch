import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm
from data import fMRIData, fMRIDataEval, LUNA, LUNAEval
from operators import MRI, Operator, CT
from torch import nn
from torch.utils.data import DataLoader
from networks import ResNet, ConvNet, ConvenientModel
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
from utils import Norm, mkdir
import json


def get_training_data(gt):
    y = operator.add_noise(operator.forward_torch(gt), NOISE_LEVEL)
    x_0 = operator.inverse_torch(y)
    # potentially add a solution of variational problem at this point
    if INITIAL_MINIMIZATION:
        x_0 = solve_variational_problem(x_0, y, 10, .1, 0, tracking=None)
    return x_0, y

def l2(x):
    return torch.mean(torch.sqrt(torch.sum(x ** 2, dim=(1,2,3))))

def get_optimal_lambda():
    gt = next(iter(data)).cuda()
    _, y = get_training_data(gt)
    norm_data = dual(operator.adjoint_torch(operator.forward_torch(gt) - y))
    return norm_data.cpu().numpy()


def solve_variational_problem(x_0, y, n_steps, step_size, lam, tracking=None, x_gt=None, global_step=None):
    '''
    Solves the variational problem starting at x_0, with data term y, n_steps
    descent steps and regularisation paremeter mu.
    param tracking: Supports values None (no tracking), Variational (writes full tracking history into seperate dir)
    and 'BestOnly' (writes best value over trajectory to training logger)
    '''
    if tracking == 'Variational':
        p = BASE_PATH + f'{EXPERIMENT}/Logs/Step_{global_step}/Lambda_{lam}/'
        print('Writing variational minimization to directory', p)
        local_writer = SummaryWriter(p)

    def add_scalar(name, value, iteration):
        if tracking == 'Variational':
            local_writer.add_scalar('Variational/' + name, value, iteration)

    def add_image(name, value, iteration):
        if tracking == 'Variational':
            value = torch.clamp(value, 0, 1)
            if value.shape[0] < N_IMAGES_LOGGING:
                v = value.cpu().numpy()
            else:
                v = (value[:N_IMAGES_LOGGING]).cpu().numpy()
            local_writer.add_images('Variational/' + name, v, iteration)

    best_per = -1
    best_attained_at = -1
    best_recon = None
    x = x_0.detach()
    add_image('Ground Truth', x_gt, 0)
    for k in range(n_steps):
        data_term = operator.forward_torch(x) - y
        add_scalar('Data_Term', l2(data_term).detach().cpu().numpy()/2, k)
        data_grad = operator.adjoint_torch(data_term)
        add_scalar('L2_Data_Gradient', l2(data_grad).detach().cpu().numpy(), k)
        add_scalar('Dual_Norm_Data_Gradient', torch.mean(dual(data_grad)).detach().cpu().numpy(), k)
        if tracking == 'Variational':
            quality = (l2(x_gt - x)).mean().detach().cpu().numpy()
            add_scalar('Quality', quality, k)
            add_image('Reconstruction', x, k)
        if not lam == 0:
            reg_grad =  regulariser.gradient(x)
            add_scalar('L2_Regulariser_Gradient', l2(reg_grad).detach().cpu().numpy(), k)
            add_scalar('Dual_Norm_Regulariser_Gradient', torch.mean(dual(reg_grad)).detach().cpu().numpy(), k)
            x = x - step_size * (data_grad + lam * reg_grad)
        else:
            x = x - step_size * data_grad
        if THRESHOLDING:
            x = torch.clamp(x, 0, 1)
        if tracking == 'BestOnly':
            quality = (l2(x_gt - x)).mean().detach().cpu().numpy()
            if quality < best_per or best_per == -1:
                best_attained_at = k
                best_per = quality
                best_recon = torch.clamp(x, 0, 1).detach().cpu().numpy()[0, ...]
    if tracking == 'BestOnly':
        tracker.add_scalar('Network_Training/Reconstruction_Quality', best_per, global_step)
        tracker.add_scalar('Network_Training/Best_Attained_At', best_attained_at, global_step)
        tracker.add_image('Network_Training/Reconstruction', best_recon, global_step)
        tracker.add_image('Network_Training/Ground Truth', torch.clamp(x_gt[0, ...], 0, 1).cpu().numpy(), global_step)
        tracker.add_image('Network_Training/FBP', torch.clamp(x_0[0, ...], 0, 1).cpu().numpy(), global_step)
    if tracking == 'BestOnly':
        return x, best_per
    else:
        return x


def gradient_penalty(gt, fbp, tracking=False, global_step=None):
    batch_size = gt.size()[0]

    # Calculate interpolation
    alpha = torch.rand(batch_size, 1, 1, 1).cuda()
    alpha = alpha.expand_as(gt)
    interpolated = Variable(alpha * gt + (1 - alpha) * fbp, requires_grad=True)

    # Calculate probability of interpolated examples
    prob_interpolated = torch.sum(regulariser(interpolated))

    # Calculate gradients of probabilities with respect to examples
    gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                           create_graph=True, retain_graph=True)[0]

    gradients_norm =  dual(gradients)
    assert len(gradients_norm.shape) == 1
    penalty = ((torch.nn.ReLU()(gradients_norm - 1)) ** 2).mean()
    if tracking:
        tracker.add_scalar('Network_Training/Gradient_Norm', gradients_norm.mean().detach().cpu().numpy(), global_step)
        tracker.add_scalar('Network_Training/Gradient_Penalty', penalty.detach().cpu().numpy(), global_step)

    # Return gradient penalty
    return penalty



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--Base", default='/store/CCIMI/sl767/Experiments/', type=str)
    parser.add_argument("--Experiment", type=str)
    args = parser.parse_args()

    BASE_PATH = args.Base
    EXPERIMENT = args.Experiment

    print('Running Experiment', EXPERIMENT)

    # load json
    with open(BASE_PATH + EXPERIMENT + '/param.json', "r") as read_file:
        parameters = json.load(read_file)


    # The following two lines are not a style guide
    for k, v in parameters.items():
        exec (k + '=v')
        print(k, v)

    norm = Norm(s=SOBOLEV, c=1 / PIXEL_SIZE)
    dual = norm.dual

    tracker = SummaryWriter(BASE_PATH + f'{EXPERIMENT}/Logs/')
    if MODALITY == 'MRI':
        operator = MRI(n_directions=N_ANGLES)
    elif MODALITY == 'CT':
        operator = CT(n_angles=N_ANGLES)
    regulariser = ResNet(channels=CHANNELS, downsamples=DOWNSAMPLING, base_path=BASE_PATH, exp_name=EXPERIMENT).cuda()
    if DATASET == 'fMRI':
        data = DataLoader(fMRIData(), batch_size=BATCH_SIZE, num_workers=4)
        val_data = DataLoader(fMRIDataEval(), batch_size=BATCH_SIZE)
    elif DATASET == 'LUNA':
        data = DataLoader(LUNA(), batch_size=BATCH_SIZE, num_workers=4)
        val_data = DataLoader(LUNAEval(), batch_size=BATCH_SIZE)
    optimizer = optim.Adam(regulariser.parameters(), lr=LEARNING_RATE)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=DECAY_FACTOR)

    global_step = regulariser.load()
    val_gt = next(iter(val_data)).cuda()
    val_fbp, val_y = get_training_data(val_gt)

    pbar = tqdm(total=TRAINING_STEPS - global_step)
    while True:
        for i, gt in enumerate(data):
            if global_step >= TRAINING_STEPS:
                regulariser.save(global_step)
                break
            gt = gt.cuda()
            # get training data
            fbp, y = get_training_data(gt)
            # compute the training loss
            loss = (regulariser(gt) - regulariser(fbp))/AVERAGE_NOISE_NORM
            r = gradient_penalty(gt, fbp, tracking=True, global_step=global_step)
            overall_loss = (loss + MU * r).mean()
            tracker.add_scalar('Network_Training/Distributional_Distance', loss.mean().detach().cpu().numpy(),
                               global_step)
            tracker.add_scalar('Network_Training/Training_Loss', overall_loss.detach().cpu().numpy(), global_step)
            tracker.add_scalar('Network_Training/Learning_Rate', optimizer.param_groups[0]['lr'], global_step)
            # update network parameters
            overall_loss.backward()
            optimizer.step()
            regulariser.zero_grad()

            # solve the variational problem
            if i % TRACKING_FREQ == 0:
                _, per = solve_variational_problem(val_fbp, val_y, n_steps=N_STEPS, step_size=STEP_SIZE, lam=LAMBDA, x_gt=val_gt,
                                          global_step=global_step, tracking='BestOnly')
                regulariser.save(global_step, performance=per)

            if (i % DECAY_EVERY_NSTEPS) == (DECAY_EVERY_NSTEPS-1):
                lr_scheduler.step()

            if i % 5001 == 5000:
                regulariser.save(global_step)

            global_step += 1
            pbar.update(1)
        if global_step >= TRAINING_STEPS:
            break
    pbar.close()
    gt = next(iter(data)).cuda()
    fbp, y = get_training_data(gt)
    for l in [LAMBDA / 3, LAMBDA, LAMBDA * 3]:
        solve_variational_problem(val_fbp, val_y, n_steps=N_STEPS, step_size=STEP_SIZE, lam=l, x_gt=val_gt,
                                  global_step=global_step, tracking='Variational')