import os
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision.utils import make_grid, save_image
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from functools import reduce

from model.loss import *
from utils.metric import *
from utils.summary import summary
import utils.visualize as vis
import config



def worker_init_fn(worker_id):        
    process_seed = torch.initial_seed()
    # Back out the base_seed so we can use all the bits.
    base_seed = process_seed - worker_id
    ss = np.random.SeedSequence([worker_id, base_seed])
    # More than 128 bits (4 32-bit words) would be overkill.
    np.random.seed(ss.generate_state(4))

def run_train(args, path_run, resume = False):    
    global global_training_step, debug    
    global_training_step = 0
    debug = args.debug
    if debug:
        global path_debug
        path_debug = os.path.join(path_run, "debug")

    dir_run = os.path.basename(path_run)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    writer = SummaryWriter(path_run) 

    train_loader = DataLoader(config.trainset, batch_size=config.training_batch_size, shuffle=True, num_workers=16, pin_memory = False, worker_init_fn=worker_init_fn)
    test_loaders = {name:DataLoader(testset, batch_size=config.test_batch_size, shuffle=False, num_workers=16, pin_memory = False) for name, testset in config.testsets.items()}

    train_lossfunc = config.test_lossfunc           
    test_lossfunc = config.test_lossfunc  
    
    model = config.model.to(device)
    optimizer = config.optimizer
    scheduler = config.scheduler
    
    
    print('Total Model Params = {:.6f}MB'.format(sum(np.prod(v.size()) for name, v in model.named_parameters() if "aux" not in name)/1e6))
    

    scaler = torch.cuda.amp.GradScaler()
    epoch_start = 0
    
    if resume:
        fullpath_model = os.path.join(path_run, "latest.pt")
        if os.path.exists(fullpath_model):
            checkpoint = torch.load(fullpath_model)
            model.load_state_dict(checkpoint['model_state_dict'])
            if not config.reset_optimizer:
                optimizer.load_state_dict(checkpoint['optimizer'])
                if args.lr != 0:
                    optimizer.param_groups[0]['lr'] = args.lr
                if scheduler:
                    scheduler.load_state_dict(checkpoint['scheduler'])            
            scaler.load_state_dict(checkpoint["scaler"])
            epoch_start = checkpoint['epoch'] + 1
    elif args.trained_model:        
        checkpoint = torch.load(args.trained_model)
        model.load_state_dict(checkpoint['model_state_dict'])

    if scheduler:
        batch_scheduler = isinstance(scheduler, (torch.optim.lr_scheduler.OneCycleLR, torch.optim.lr_scheduler.CyclicLR, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts))
        epoch_scheduler = not batch_scheduler
    else:
        batch_scheduler = None
        epoch_scheduler = None

    train_loss_min = np.inf    
    test_loss_mins = {name:np.inf for name in config.testsets}
    for epoch in range(epoch_start, config.epoch):   
        train_loss = train(epoch, model, train_loader, optimizer, scaler, scheduler, batch_scheduler, train_lossfunc, 
                            writer, device, imglog_period = config.imglog_period)        
        test_losses = {name:test(epoch, model, name, test_loaders[name], test_lossfunc, 
                            writer, device, imglog_period = config.imglog_period) for name in config.testsets}
        if epoch_scheduler:
            scheduler.step()

        save_model(epoch, model, optimizer, scaler, scheduler, train_loss, test_losses, path_run, "latest.pt")
        if train_loss < train_loss_min:
            save_model(epoch, model, optimizer, scaler, scheduler, train_loss, test_losses, path_run, "best_train.pt")
            train_loss_min = train_loss
        for name in config.testsets:            
            if test_losses[name] < test_loss_mins[name]:
                save_model(epoch, model, optimizer, scaler, scheduler, train_loss, test_losses, path_run, "best_"+name+".pt")
                test_loss_mins[name] = test_losses[name]

        print("{:}, Epoch {:}, lr {:.2e}, Train Loss {:.3f}, Mem usage {:.3f}gb".format(
                dir_run, epoch, optimizer.param_groups[0]['lr'], train_loss, torch.cuda.max_memory_reserved()/10**9))
        print()        
        
    writer.flush()
    writer.close()

    while True:
        pass


def train(epoch, model, loader, optimizer, scaler, scheduler, batch_scheduler, lossfunc, writer, device, imglog_period = 1):
    global global_training_step, debug
    model.train()    
    total_loss = 0
    total_err = 0
    loop_batch = tqdm(loader)
    num_samples = 0
    np.random.seed()
    if debug: 
        global path_debug
        prev_model = copy.deepcopy(model)
        prev_model.to('cpu')       
         

    for batch_idx, batch in enumerate(loop_batch):
        global_training_step += 1
        left_, right_, gt_, mask_ = batch 
        left, right, gt = left_.to(device), right_.to(device), gt_.to(device)
        with torch.cuda.amp.autocast(enabled = config.use_amp):
            start = time.time()
            out, auxiliary = model(left, right, d_max = config.d_max_training_model)
            end = time.time()  
            elapsed = end-start
            valid = mask_
            valid[gt_ >= config.d_max_training_eval] = False        
            valid[gt_ < 0] = False            
            loss = lossfunc(out, gt, auxiliary, valid, d_max = config.d_max_training_eval)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = config.clip_grad_max_norm)
        scaler.step(optimizer)
        scale_before = scaler.get_scale()
        scaler.update()

        if debug:
            if torch.isnan(loss):
                print(path_debug)
                os.makedirs(path_debug, exist_ok=True)
                with open(os.path.join(path_debug, "debug_current.txt"), 'w') as file:
                    summary(model, file)
                with open(os.path.join(path_debug, "debug_prev.txt"), 'w') as file:
                    summary(prev_model, file)
                checkpoint = {
                    "batch" : batch,
                    "prev_model" : prev_model.state_dict(),
                    "current_model" : model.state_dict(),
                }
                torch.save(checkpoint, os.path.join(path_debug, "debug.pt"))

                raise Exception("Loss is NaN")                
            prev_model = copy.deepcopy(model)
            prev_model.to('cpu')
            
        optimizer.zero_grad()  
        diff = (out - gt).abs()    
        err = sum(get_epe(diff, valid))

        if batch_scheduler and scale_before > scaler.get_scale() and (not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR) or scheduler.last_epoch < scheduler.total_steps):        
            scheduler.step()
        total_loss += loss.item() * len(gt)
        total_err += err
        num_samples += len(gt)        
        loop_batch.set_description("Epoch {}, lr {:.2e}, Loss {:.3f}, Err {:.3f}, time {:.3f}, Mem {:.3f}gb".format(epoch, optimizer.param_groups[0]['lr'], loss.item(), total_err/num_samples, elapsed, torch.cuda.max_memory_reserved()/10**9))
    
    total_loss = total_loss / num_samples
    total_err = total_err / num_samples
    writer.add_scalar('train/loss', total_loss, epoch)
    writer.add_scalar('train/EPE', total_err, epoch)
    if epoch % imglog_period == 0:                 
        log_image(writer, 'train', epoch, left_[0], right_[0], mask_[0], gt_[0], out[0].cpu().detach())
    
    return total_loss


def test(epoch, model, name, loader, lossfunc, writer, device, imglog_period = 1):
    model.eval()
    
    total_loss = 0
    total_err = 0
    total_bads = {th:0 for th in config.test_bad_ths}
    if config.test_D1: total_D1 = 0
    loop_batch = tqdm(loader)
    num_samples = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loop_batch):
            left_, right_, gt_, mask_ = batch 
            left, right, gt = left_.to(device), right_.to(device), gt_.to(device)            
            start = time.time()
            out, auxiliary = model(left, right, d_max = config.d_max_test_models[name])
            end = time.time()  
            elapsed = end-start
            valid = mask_
            valid[gt_ >= config.d_max_test_evals[name]] = False        
            valid[gt_ < 0] = False
            loss = lossfunc(out, gt, auxiliary, valid, d_max = config.d_max_test_evals[name])            
            diff = (out - gt).abs() 
            err = sum(get_epe(diff, valid))            
            bads = {th:sum(get_bad(diff, th, valid)) for th in config.test_bad_ths}
            d1 = sum(get_D1(diff, gt, valid))

            total_loss += loss.item() * len(gt)
            total_err += err            
            total_bads = {th: total_bads[th] + bads[th] for th in config.test_bad_ths}
            if config.test_D1: total_D1 += d1
            num_samples += len(gt)
            loop_batch.set_description("Epoch {}, Err {:.3f}, time {:.3f}, Mem {:.3f}gb".format(epoch, total_err/num_samples, elapsed, torch.cuda.max_memory_reserved()/10**9))

    total_loss = total_loss / num_samples
    total_err = total_err / num_samples
    total_bads = {th: total_bads[th] / num_samples for th in config.test_bad_ths}
    writer.add_scalar(name+'/EPE', total_err, epoch)
    for th in config.test_bad_ths:
        writer.add_scalar(name+f'/bad{th}', total_bads[th], epoch)
    if config.test_D1: 
        total_D1 = total_D1 / num_samples
        writer.add_scalar(name+f'/D1', total_D1, epoch)            
    if epoch % imglog_period == 0:
        log_image(writer, name, epoch, left_[0], right_[0], mask_[0], gt_[0], out[0].cpu().detach())

    return total_loss



def log_image(writer, name, global_step:int, img_left, img_right, mask, gt, out, max_disp = 192):
    err = (gt - out).abs() * mask
    out_cmap = vis.apply_colomap(out[0], range=(0, max_disp), colormap='turbo', is_torch=True)
    gt_cmap = vis.apply_colomap(gt[0], range=(0, max_disp), colormap='turbo', is_torch=True)
    err_cmap = vis.apply_colomap(err[0], range=(0, int(0.1*max_disp)), colormap='inferno', is_torch=True)
    img_res = make_grid([out_cmap, gt_cmap, err_cmap], padding=0)
    img_in = make_grid([img_left/255., img_right/255., torch.tile(mask, (3,1,1))], padding=0)
    writer.add_images(name+'_result', img_res[None], global_step)
    writer.add_images(name+'_input', img_in[None], global_step)        
    


def save_model(epoch, model, optimizer, scaler, scheduler, train_loss, test_losses, path_log, fname_model):
    fullpath_model = os.path.join(path_log, fname_model)
    fullpath_log = os.path.join(path_log, "log.txt")
    dict_log = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),                    
                    'scaler': scaler.state_dict(),
                    'train_loss': train_loss,
                }
    dict_log.update(test_losses)

    if scheduler:
        dict_log['scheduler'] = scheduler.state_dict()
    torch.save(dict_log, fullpath_model)
    if not os.path.exists(fullpath_log):
        with open(fullpath_log, 'a') as f:            
            str_tests = reduce(lambda a,b: a+', '+b, test_losses.keys())
            f.write('date and time, epoch, train_loss, ' + str_tests + ', file \n')
            f.write('--------------------------------------------------------------------\n')
    with open(fullpath_log, 'a') as f:
        str_test_losses = ['{:.8f}'.format(a) for a in test_losses.values()]
        str_test_losses = reduce(lambda a,b: a + ', ' + b, str_test_losses)
        f.write('{:}, {:3d}, {:.8f}, {:}, {:}\n'.format(time.ctime(), epoch, train_loss, str_test_losses, fullpath_model))