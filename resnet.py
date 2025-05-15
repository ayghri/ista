import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import os
import time
from tqdm import tqdm  # For progress bars
import models

# ==============================================================================
# Training and Testing Functions
# ==============================================================================


# Training function
def train(epoch, model, trainloader, optimizer, criterion, device, scheduler):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    pbar = tqdm(trainloader, desc=f"Epoch {epoch+1} Training", leave=True)
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        pbar.set_postfix(
            Loss=f"{train_loss/(batch_idx+1):.3f}",
            Acc=f"{100.*correct/total:.3f}%",
            LR=f'{optimizer.param_groups[0]["lr"]:.4f}',
        )

    # Step the scheduler after each epoch
    scheduler.step()


# Testing function
def test(epoch, model, testloader, criterion, device, best_acc, model_name):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(testloader, desc=f"Epoch {epoch+1} Testing ", leave=False)
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix(
                Loss=f"{test_loss/(batch_idx+1):.3f}",
                Acc=f"{100.*correct/total:.3f}%",
            )

    # Save checkpoint.
    acc = 100.0 * correct / total
    print(f"\nEpoch {epoch+1}: Test Acc: {acc:.3f}%")
    if acc > best_acc[0]:  # Use list to modify in-place
        print(f"Saving Best Model... (Acc: {acc:.3f}%)")
        state = {
            "net": model.state_dict(),
            "acc": acc,
            "epoch": epoch,
        }
        if not os.path.isdir("checkpoint"):
            os.mkdir("checkpoint")
        torch.save(state, f"./checkpoint/{model_name}_best.pth")
        best_acc[0] = acc  # Update best accuracy
    return acc


# ==============================================================================
# Main Execution
# ==============================================================================
if __name__ == "__main__":
    # --- Configuration ---
    TARGET_SPARSITY = 0.1
    DYSTIL_BETA = 0.01
    MODEL_NAME = f"ResNet50_CIFAR10_reduced{TARGET_SPARSITY}"  # Choose ResNet18, ResNet34, ResNet50 etc.
    DATA_DIR = "./data_cifar10"
    CHECKPOINT_DIR = "./checkpoint"
    EPOCHS = 200  # Needs enough epochs
    # Standard starting LR for SGD
    # INITIAL_LR = 0.1 * (1 / TARGET_SPARSITY) ** 0.5 / 1.4
    INITIAL_LR = 0.1
    BATCH_SIZE = 128
    MOMENTUM = 0.9
    # WEIGHT_DECAY = 5e-4 * TARGET_SPARSITY  # Crucial regularization
    # WEIGHT_DECAY = 0.0  # Crucial regularization
    WEIGHT_DECAY = 5e-4  # Crucial regularization
    NUM_WORKERS = 4  # Adjust based on your system

    # --- Device Setup ---
    # if torch.backends.mps.is_available():
    #     device = torch.device("mps")
    #     print("Using device: MPS (Apple Silicon GPU)")
    # elif torch.cuda.is_available():
    #     print("Using device: CUDA (NVIDIA GPU)")
    #     cudnn.benchmark = (
    #         True  # Enable cuDNN benchmark mode for potential speedup
    #     )
    # else:
    #     device = torch.device("cpu")
    #     print("Warning: Using CPU.")

    device = torch.device("cuda")

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    # --- Data ---
    print("==> Preparing data..")
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2023, 0.1994, 0.2010)

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=transform_train
    )
    trainloader = DataLoader(
        trainset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True if device != "cpu" else False,
        persistent_workers=True if NUM_WORKERS > 0 else False,
    )

    testset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=transform_test
    )
    testloader = DataLoader(
        testset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if device != "cpu" else False,
        persistent_workers=True if NUM_WORKERS > 0 else False,
    )

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    # --- Model ---
    print("==> Building model..")
    # Select model based on configuration
    if MODEL_NAME.startswith("ResNet18"):
        net = models.ResNet18()
    elif MODEL_NAME.startswith("ResNet34"):
        net = models.ResNet34()
    elif MODEL_NAME.startswith("ResNet50"):
        net = models.ResNet50()
    elif MODEL_NAME.startswith("ResNet101"):
        net = models.ResNet101()
    elif MODEL_NAME.startswith("ResNet152"):
        net = models.ResNet152()
    else:
        raise ValueError(f"Unsupported model name: {MODEL_NAME}")

    net = net.to(device)
    # Optional: Use multiple GPUs if available
    # if device == 'cuda' and torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs!")
    #     net = nn.DataParallel(net)

    # --- Loss, Optimizer, Scheduler ---
    criterion = nn.CrossEntropyLoss()
    from sgd import DySTiLSGD

    def set_sparsity(model, ratio):
        p_total = 0
        p_used = 0
        dystil_exclude = ["conv1.weight"]
        dystil_pattern = ["bn", "bias"]
        for name, p in model.named_parameters():
            p_total += p.numel()
            if p.ndim == 1:
                continue
            if not (
                any([ptrn in name for ptrn in dystil_pattern])
                or name in dystil_exclude
            ):
                p.dystil_k = int(p.numel() * ratio)
                p.dystil_beta = DYSTIL_BETA
                print(name, p.numel(), p.shape, p.dystil_k, p.dystil_beta)
                p_used += p.dystil_k
            else:
                p_used += p.numel()

        print("Target Sparsity:", 1 - p_used / p_total)

    from tutils import (
        initialize_sparse,
        calculate_sparsity_per_layer,
    )

    set_sparsity(net, TARGET_SPARSITY)
    initialize_sparse(net)

    optimizer = DySTiLSGD(
        net.parameters(),
        lr=INITIAL_LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )
    # Use Cosine Annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )

    # --- Training Loop ---
    best_acc = [0.0]  # Use list to allow modification inside 'test' function
    start_epoch = 0  # Can be modified if resuming from checkpoint

    print(f"\n==> Starting Training for {EPOCHS} epochs...")
    start_time_train = time.time()

    for epoch in range(start_epoch, start_epoch + EPOCHS):
        epoch_start_time = time.time()
        train(epoch, net, trainloader, optimizer, criterion, device, scheduler)
        current_acc = test(
            epoch, net, testloader, criterion, device, best_acc, MODEL_NAME
        )
        epoch_duration = time.time() - epoch_start_time
        print(
            f"Epoch {epoch+1} Duration: {epoch_duration:.2f}s - Best Acc: {best_acc[0]:.3f}%"
        )

        overall_sparse, _ = calculate_sparsity_per_layer(net)

        print(f"Overall sparsity: {overall_sparse:.2f}%")
        print("-" * 60)

    total_training_time = time.time() - start_time_train
    print(f"\nTraining finished in {total_training_time / 3600:.2f} hours.")
    print(f"Best Test Accuracy achieved: {best_acc[0]:.3f}%")
    print(f"Best model saved to ./checkpoint/{MODEL_NAME}_best.pth")
