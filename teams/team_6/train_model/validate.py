import torch
import torch.nn as nn
import config
from tqdm import tqdm
import torchmetrics


def valid(
    model,
    dataloader,
):
    
    training_loss = 0.0
    num_correct = 0
    num_example = 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Training"):
            image = batch["img"].to(device=config.DEVICE)
            targets1 = batch["target1"].to(device=config.DEVICE)
            targets2 = batch["target2"].to(device=config.DEVICE)
            targets3 = batch["target3"].to(device=config.DEVICE)
            all_targets = torch.stack([targets1, targets2, targets3], dim=1)
    
            all_outputs = model(image)
            loss = nn.BCEWithLogitsLoss(weight=None, size_average=None, reduce=None, reduction='mean')(all_outputs.float(), all_targets.float())
            
            training_loss += loss.data.item()
            num_correct += (
                (
                    (all_outputs > 0.5) == all_targets
                ).sum().item())
            
            num_example += all_targets.size(0) * 3
            
    
        training_loss /= len(dataloader.dataset)
        accuracy = num_correct / num_example
    
        return training_loss, accuracy, 0