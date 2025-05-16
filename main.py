import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn


import os
import time

# from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
from train_utils import check_wandb_login, train_one_epoch
from evaluate import evaluate
from datasets import get_dataset
import wandb


@hydra.main(config_path="config", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    # check_wandb_login()
    # wandb.init(
    #     project=cfg.wandb.project,
    #     entity=cfg.wandb.get("entity", None),
    #     config=OmegaConf.to_container(cfg, resolve=True),
    #     mode=cfg.wandb.mode,
    # )

    # Accessing your configuration
    # hidden_size = cfg.experiment.model

    # Log configuration for debugging
    print(OmegaConf.to_yaml(cfg))

    # for epoch in range(epochs):
    # loss = train_one_epoch(lr, batch_size, hidden_size)

    # Logging metrics to wandb
    # wandb.log({"epoch": epoch, "loss": loss})
    exp = cfg.experiment
    epochs = exp.epochs
    lr = exp.lr
    batch_size = exp.batch_size

    # --- Device Setup ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: CUDA (NVIDIA GPU)")
        cudnn.benchmark = (
            True  # Enable cuDNN benchmark mode for potential speedup
        )
    else:
        device = torch.device("cpu")
        print("Warning: Using CPU.")

    if not os.path.exists(cfg.data_dir):
        os.makedirs(cfg.data_dir)
    if not os.path.exists(cfg.checkpoint_dir):
        os.makedirs(cfg.checkpoint_dir)

    # --- Data ---

    # --- Model ---
    print("==> Building model..")
    # Select model based on configuration
    from models import get_model

    net = get_model(exp.model)

    net = net.to(device)
    # Optional: Use multiple GPUs if available
    # if device == 'cuda' and torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs!")
    #     net = nn.DataParallel(net)

    # --- Loss, Optimizer, Scheduler ---
    criterion = nn.CrossEntropyLoss()
    if exp.method == "dystil":
        from optimizers.dystil import DySTiLSGD as Optimizer
        from train_utils import set_dystil_params

        set_dystil_params(net, 1 - exp.sparsity, exp.beta)
    elif exp.method == "iht":
        from optimizers.iht import IHTSGD as Optimizer
    else:
        raise ValueError("Unknown method:", exp.method)

    from tutils import (
        initialize_sparse,
        calculate_sparsity_per_layer,
    )

    initialize_sparse(net)

    optimizer = Optimizer(
        net.parameters(),
        lr=lr,
        momentum=exp.momentum,
        weight_decay=exp.weight_decay,
    )
    # Use Cosine Annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.experiment.t_max
    )

    # --- Training Loop ---
    best_acc = [0.0]  # Use list to allow modification inside 'test' function
    start_epoch = 0  # Can be modified if resuming from checkpoint

    print(f"\n==> Starting Training for {epochs} epochs...")
    start_time_train = time.time()

    trainloader, testloader = get_dataset(
        cfg.dataset.name,
        cfg.data_dir,
        batch_size,
        device=device,
        num_workers=exp.num_workers,
    )

    ckpt_name = (f"{cfg.dataset.name}_{exp.model}",)

    for epoch in range(start_epoch, start_epoch + epochs):
        epoch_start_time = time.time()
        train_one_epoch(
            epoch, net, trainloader, optimizer, criterion, device, scheduler
        )

        current_acc = evaluate(
            epoch, net, testloader, criterion, device, best_acc, ckpt_name
        )

        # Save checkpoint.
        # print(f"\nEpoch {epoch+1}: Test Acc: {acc:.3f}%")
        # if acc > best_acc[0]:  # Use list to modify in-place
        #     print(f"Saving Best Model... (Acc: {acc:.3f}%)")
        #     best_acc[0] = acc  # Update best accuracy
        # return acc
        epoch_duration = time.time() - epoch_start_time
        print(
            f"Epoch {epoch+1}/{epochs} Duration: {epoch_duration:.2f}s - Best Acc: {best_acc[0]:.3f}%"
        )

        overall_sparse, _ = calculate_sparsity_per_layer(net)

        print(f"Overall sparsity: {overall_sparse:.2f}%")
        print("-" * 60)

    total_training_time = time.time() - start_time_train
    print(f"\nTraining finished in {total_training_time / 3600:.2f} hours.")
    print(f"Best Test Accuracy achieved: {best_acc[0]:.3f}%")
    print(f"Best model saved to ./checkpoint/{ckpt_name}_?.pth")

    # wandb.finish()


if __name__ == "__main__":
    main()
