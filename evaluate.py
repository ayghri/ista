import torch
import os


# Testing function
@torch.no_grad()
def evaluate(epoch, model, loader, criterion, device, best_acc, model_name):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    test_loss = test_loss / total
    acc = 100.0 * correct / total
    return acc, test_loss

    # Save checkpoint.
    print(f"\nEpoch {epoch+1}: Test Acc: {acc:.3f}%")
    if acc > best_acc[0]:  # Use list to modify in-place
        print(f"Saving Best Model... (Acc: {acc:.3f}%)")
        best_acc[0] = acc  # Update best accuracy
    return acc

