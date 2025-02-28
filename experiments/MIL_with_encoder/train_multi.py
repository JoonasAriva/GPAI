from __future__ import print_function

import logging
import os
import sys
sys.path.append('/gpfs/space/home/joonas97/GPAI/')
sys.path.append('/users/arivajoo/GPAI')
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.optim as optim
import wandb
from omegaconf import OmegaConf, DictConfig
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from data.kidney_dataloader import KidneyDataloader
from depth_trainer import DepthLossV2
import torch.utils.data as data_utils
# from torchinfo import summary
from torch.utils.data.distributed import DistributedSampler


from train_utils import prepare_dataloader, pick_model, pick_trainer, prepare_optimizer
from tqdm import tqdm
import time
import gc
from contextlib import nullcontext

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# Training settings

dir_checkpoint = Path('./checkpoints/')

np.seterr(divide='ignore', invalid='ignore')

os.environ["WANDB__SERVICE_WAIT"] = "300"


def print_slurm_env():
    slurm_env = "\n".join(
        [
            "=" * 80,
            f"SLURM Process: {os.environ.get('SLURM_PROCID', 'N/A')=}",
            "=" * 80,
            f"{os.environ.get('SLURM_NTASKS', 'N/A')=}",
            f"{os.environ.get('SLURM_LOCALID', 'N/A')=}",
            f"{os.environ.get('RANK', 'N/A')=}",
            f"{os.environ.get('LOCAL_RANK', 'N/A')=}",
            f"{os.environ.get('WORLD_SIZE', 'N/A')=}",
            f"{os.environ.get('MASTER_ADDR', 'N/A')=}",
            f"{os.environ.get('MASTER_PORT', 'N/A')=}",
            f"{os.environ.get('ROCR_VISIBLE_DEVICES', 'N/A')=}",
            f"{os.environ.get('SLURM_JOB_GPUS', 'N/A')=}",
            f"{os.sched_getaffinity(0)=}",
            f"{os.environ.get('TORCH_NCCL_ASYNC_ERROR_HANDLING', 'N/A')=}",
            "-" * 80 + "\n",
        ]
    )
    print(slurm_env, flush=True)


@hydra.main(config_path="config", config_name="config", version_base='1.1')
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    print(f"Running {cfg.experiment}, Work in {os.getcwd()}")

    np.random.seed(cfg.training.seed)
    torch.manual_seed(cfg.training.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.training.seed)
        print('\nGPU is ON!')

    if cfg.training.multi_gpu:
        # torch.multiprocessing.set_start_method("spawn")
        # local_rank = int(os.environ['LOCAL_RANK'])
        init_process_group(backend="nccl")
    print_slurm_env()
    print('Load Train and Test Set')


    dataloader_params = {
        'only_every_nth_slice': 1, 'as_rgb': False,
        'plane': 'axial', 'center_crop': cfg.data.crop_size, 'downsample': False,
        'roll_slices': False, 'model_type': '3D',
        'generate_spheres': False, 'patchify': False,
        'no_lungs': False}
    train_dataset = KidneyDataloader(type="train",
                                     augmentations=None,
                                     **dataloader_params, random_experiment=cfg.data.random_experiment)
    test_dataset = KidneyDataloader(type="test", **dataloader_params)

    loader_kwargs = {'num_workers': 7, 'pin_memory': True} if torch.cuda.is_available() else {}


    print("creating distributed sampler")
    print("rank in train utils: ", int(os.environ["LOCAL_RANK"]))
    sampler = DistributedSampler(train_dataset, num_replicas=2, rank=int(os.environ["LOCAL_RANK"]), shuffle=True)
    train_loader = data_utils.DataLoader(train_dataset, batch_size=cfg.data.batch_size, sampler=sampler,
                                         **loader_kwargs)

    test_loader = data_utils.DataLoader(test_dataset, batch_size=cfg.data.batch_size, shuffle=False,
                                        **loader_kwargs)
    steps_with_parnu = 1250
    steps_in_epoch = steps_with_parnu // cfg.data.batch_size  # before pärnu it was 500
    if cfg.data.dataloader == "kidney_real":
        proj_name = "MIL_encoder_24"
    elif cfg.data.dataloader == "synthetic" or "kidney_synth":
        proj_name = "MIL_encoder_synth24"
    if cfg.experiment == 'depth':
        if cfg.data.dataloader == "abdomen_atlas":
            proj_name = "abdomen_atlas_depth"
            steps_in_epoch = 4666
        else:
            proj_name = "depth"
    if cfg.experiment == 'swin':
        proj_name = "swin_compass"

    logging.info('Init Model')
    model = pick_model(cfg)



    if cfg.training.multi_gpu == True:
        n_gpus = torch.cuda.device_count()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        model.cuda()
        print("Using {} GPUs".format(n_gpus))
        print("Local rank: {}".format(local_rank))
        model = DDP(  # <- We need to wrap the model with DDP
            model,
            device_ids=[local_rank],  # <- and specify the device_ids/output_device
        )

    optimizer = prepare_optimizer(cfg, model)

    number_of_epochs = cfg.training.epochs
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, total_steps=int(
        number_of_epochs * steps_in_epoch / cfg.training.weight_update_freq),
                                              pct_start=0.2, max_lr=[cfg.training.learning_rate,
                                                                     cfg.training.learning_rate * 10] if cfg.experiment == "compass_twostage" else cfg.training.learning_rate)

    if not cfg.check:
        experiment = wandb.init(project=proj_name, anonymous='must')

        experiment.config.update(
            dict(epochs=cfg.training.epochs,
                 learning_rate=cfg.training.learning_rate, model_name=cfg.model.name,
                 weight_decay=cfg.training.weight_decay, config=cfg))
        logging.info('Start Training')

    best_loss = 10000
    best_f1 = 0
    best_epoch = 0
    not_improved_epochs = 0

    train = True
    loss_function = DepthLossV2(step=0.1).cuda()
    for epoch in range(1, cfg.training.epochs + 1):
        epoch_results = dict()

        results = dict()
        epoch_loss = 0.
        depth_loss = 0.
        class_loss = 0.
        step = 0
        nr_of_batches = len(train_loader)

        tepoch = tqdm(train_loader, unit="batch", ascii=True,
                      total=steps_in_epoch if steps_in_epoch > 0 and train else len(train_loader))

        if train:
            model.train()
        else:
            model.eval()
            # model.disable_dropout()

        scaler = torch.cuda.amp.GradScaler()
        optimizer.zero_grad(set_to_none=True)
        data_times = []
        forward_times = []
        backprop_times = []
        loss_times = []
        outputs = []
        targets = []

        attention_scores = dict()
        attention_scores["all_scans"] = [[], []]  # first is for all label acc, second is tumor specific
        attention_scores["cases"] = [[], []]
        attention_scores["controls"] = [[], []]
        time_data = time.time()

        for data, bag_label, meta in tepoch:

            if cfg.check:
                print("data shape: ", data.shape, flush=True)

            data_time = time.time() - time_data
            data_times.append(data_time)

            tepoch.set_description(f"Epoch {epoch}")
            step += 1
            gc.collect()

            # data = torch.permute(torch.squeeze(data), (3, 0, 1, 2)) 2.5
            #data = torch.permute(data, (0, 4, 1, 2, 3)) multi gpu 2.5


            # data = data.to(self.device, dtype=torch.float16, non_blocking=True)
            data = data.cuda(non_blocking=True).to(dtype=torch.float16)
            print("data device: ", data.device, "data shape: ", data.shape)
            print("device: ", int(os.environ["LOCAL_RANK"]))
            time_forward = time.time()
            with torch.cuda.amp.autocast(), torch.no_grad() if not train else nullcontext():

                position_scores = model.forward(data, scan_end=meta[3])

                forward_time = time.time() - time_forward
                forward_times.append(forward_time)

                time_loss = time.time()
                d_loss = loss_function(position_scores, z_spacing=meta[2][2],  # second 2 is for z spacing
                                       nth_slice=meta[1])

                loss_time = time.time() - time_loss
                loss_times.append(loss_time)
                total_loss = (d_loss)

            if train:
                time_backprop = time.time()
                scaler.scale(total_loss).backward()
                backprop_time = time.time() - time_backprop
                backprop_times.append(backprop_time)
                if (step) % 1== 0 or (step) == len(train_loader):

                    scaler.step(optimizer)
                    scaler.update()
                    if scheduler:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

            epoch_loss += total_loss.item()
            depth_loss += d_loss.item()

            if step >= 6 and cfg.check:
                break

            if train and steps_in_epoch > 0:
                if step >= steps_in_epoch:
                    nr_of_batches = steps_in_epoch
                    break

            time_data = time.time()

        # calculate loss and error for epoch

        epoch_loss /= nr_of_batches
        depth_loss /= nr_of_batches
        class_loss /= nr_of_batches

        print("data speed: ", round(np.mean(data_times), 3), "forward speed ", round(np.mean(forward_times), 3),
              "backprop speed: ", round(np.mean(backprop_times), 3), "loss speed: ", round(np.mean(loss_times), 3), )

        print(
            '{}: loss: {:.4f}'.format(
                "Train" if train else "Validation", epoch_loss))

        results["epoch"] = epoch
        results["loss"] = epoch_loss
        results["depth_loss"] = depth_loss



        train_results = {k + '_train': v for k, v in train_results.items()}
        test_results = {k + '_test': v for k, v in test_results.items()}

        test_loss = test_results["loss_test"]

        epoch_results.update(train_results)
        epoch_results.update(test_results)

        if cfg.check:
            logging.info("Model check completed")
            return

        Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)

        if test_loss < best_loss:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(dir_checkpoint / 'best_model.pth'))
            logging.info(f"Best new model at epoch {epoch} (lowest test loss: {test_loss})!")

            best_loss = test_loss
            best_epoch = epoch
            not_improved_epochs = 0

        else:
            if not_improved_epochs > 20:
                logging.info("Model has not improved for the last 20 epochs, stopping training")
                # break
            not_improved_epochs += 1

        experiment.log(epoch_results)
        torch.save(model.state_dict(), str(dir_checkpoint / 'current_model.pth'))
        torch.save(optimizer.state_dict(), str(dir_checkpoint / 'current_optimizer.pth'))
        torch.save(scheduler.state_dict(), str(dir_checkpoint / 'current_scheduler.pth'))

    torch.save(model.state_dict(), str(dir_checkpoint / 'last_model.pth'))
    logging.info(f'Last checkpoint! Checkpoint {epoch} saved!')
    logging.info(f"Training completed, best_metric (f1-test): {best_f1:.4f} at epoch: {best_epoch}")


if __name__ == "__main__":
    main()
