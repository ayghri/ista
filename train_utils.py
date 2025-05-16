import os
import sys
import wandb
from tqdm import tqdm
import torch

WANDB_API_KEY_ENV_VAR = "WANDB_API_KEY"


def check_wandb_login():
    """
    Checks for W&B login, prompts for API key if not found or set,
    and verifies the login by fetching user information.
    Exits on failure.
    """
    api_key = os.environ.get(WANDB_API_KEY_ENV_VAR, "").strip()

    if not api_key:
        print(
            f"W&B API key not found in env variable '{WANDB_API_KEY_ENV_VAR}'."
        )
        try:
            api_key = input("Paste your W&B API key here: ").strip()
            if not api_key:
                print("❌ No API key provided. Exiting.")
                sys.exit(1)
        except EOFError:
            print(
                "❌ W&B API key prompt failed (EOFError). Running in a non-interactive environment?"
            )
            print(
                f" Please set the '{WANDB_API_KEY_ENV_VAR}' env variable. Exiting."
            )
            sys.exit(1)
        os.environ[WANDB_API_KEY_ENV_VAR] = api_key
        print(f"API Key temporarily set in environment for this session.")

    print("Attempting to log into W&B...")
    try:
        success = wandb.login(key=api_key, relogin=True)
        if not success:
            masked_key = (
                f"{api_key[:4]}...{api_key[-4:]}"
                if len(api_key) > 8
                else "key_too_short_to_mask"
            )
            print(
                f"❌ W&B login failed with the provided API key (masked: {masked_key})."
            )
            print(
                "   Please ensure your API key is correct and has not expired."
            )
            sys.exit(1)
    except Exception as e:
        print(f"❌ An error occurred during W&B login: {e}")
        sys.exit(1)

    print("Verifying W&B login status via API...")
    try:
        api = wandb.Api()
        viewer = api.viewer

        username = viewer.username
        entity = viewer.entity

        if username:
            user_display = f"'{username}'"
            if (
                entity and entity != username
            ):  # Sometimes entity and username can be the same
                user_display += f" (Entity: '{entity}')"
            print(f"✅ Successfully logged into W&B as {user_display}.")
        elif (
            entity
        ):  # Fallback if username is somehow not available but entity is
            print(
                f"✅ Successfully logged into W&B (Entity: '{entity}',"
                "username not directly found in viewer)."
            )
        else:
            # This case should be rare if login succeeded and API is reachable
            print(
                "⚠️ W&B login call was successful, "
                "but could not retrieve user/entity details from API."
            )
            # You might want to exit here if strict verification is needed
            # sys.exit(1)

    except wandb.errors.CommError as e:
        print(
            "⚠️ W&B login call successful, "
            f"but API communication error during verification: {e}"
        )
        print(
            " This could be due to network issues,"
            "an invalid/revoked API key after initial check,\n"
            " or API key having insufficient permissions."
        )
        sys.exit(1)
    except Exception as e:
        print(
            "⚠️ W&B login call successful,"
            "but an unexpected error occurred during API verification: {e}"
        )
        sys.exit(1)


def train_one_epoch(
    epoch,
    model,
    trainloader,
    optimizer,
    criterion,
    device,
    scheduler=None,
    n_epochs=1,
):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    pbar = tqdm(
        trainloader, desc=f"Epoch {epoch+1}/{n_epochs} Training", leave=False
    )
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
    if scheduler is not None:
        scheduler.step()


def save_checkpoint(model, model_name, epoch, checkpoint_dir, cfg):
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    state = {
        "state_dict": model.state_dict(),
        # "acc": acc,
        "epoch": epoch,
        "cfg": cfg,
    }
    torch.save(state, f"./checkpoint/{model_name}_best_{epoch}.pth")


def set_dystil_params(model, density, beta):
    p_total = 0
    p_used = 0
    dystil_exclude = ["conv1.weight"]
    for name, p in model.named_parameters():
        p_total += p.numel()
        if not ("bias" in name or name in dystil_exclude or "bn" in name):
            p.dystil_k = int(p.numel() * density)
            p.dystil_beta = beta
            print(name, p.numel(), p.shape, p.dystil_k)
            p_used += p.dystil_k
        else:
            p_used += p.numel()

    print("Target Sparsity:", 1 - p_used / p_total)


# def set_sparsity(model, ratio):
#     p_total = 0
#     p_used = 0
#     dystil_exclude = ["conv1.weight"]
#     for name, p in model.named_parameters():
#         p_total += p.numel()
#         if not ("bias" in name or name in dystil_exclude or "bn" in name):
#             p.dystil_k = int(p.numel() * ratio)
#             p.dystil_beta = DYSTIL_BETA
#             print(name, p.numel(), p.shape, p.dystil_k)
#             p_used += p.dystil_k
#         else:
#             p_used += p.numel()
#
#     print("Target Sparsity:", 1 - p_used / p_total)


if __name__ == "__main__":

    check_wandb_login()
    print("\nScript can now proceed with W&B operations.")
