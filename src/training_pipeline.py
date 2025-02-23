import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
import utils

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    """
    Performs a single training step on the given model using the provided dataloader.
    
    Args:
        model (torch.nn.Module): The neural network model to train.
        dataloader (torch.utils.data.DataLoader): The DataLoader providing training data.
        loss_fn (torch.nn.Module): The loss function to minimize.
        optimizer (torch.optim.Optimizer): The optimizer to update model weights.
        device (torch.device): The device to run the training (CPU/GPU).
    
    Returns:
        Tuple[float, float]: The average training loss and accuracy for the epoch.
    """

    model.train()  # Set the model to training mode

    train_loss, train_acc = 0, 0  # Initialize loss and accuracy accumulators

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)  # Move data to the specified device

        # 1. Forward pass: Compute model predictions
        y_pred = model(X)

        # 2. Compute loss and accumulate it
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Zero gradients before backward pass
        optimizer.zero_grad()

        # 4. Backward pass: Compute gradients
        loss.backward()

        # 5. Update model weights using optimizer
        optimizer.step()

        # 6. Compute accuracy
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)  # Get predicted class labels
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)  # Compute batch accuracy

    # Compute average loss and accuracy over all batches
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)

    return train_loss, train_acc

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    """
    Evaluates the model on the test dataset.

    Args:
        model (torch.nn.Module): The trained model.
        dataloader (torch.utils.data.DataLoader): The DataLoader providing test data.
        loss_fn (torch.nn.Module): The loss function.
        device (torch.device): The device to run the evaluation (CPU/GPU).
    
    Returns:
        Tuple[float, float]: The average test loss and accuracy.
    """

    model.eval()  # Set model to evaluation mode (disables dropout, batch norm, etc.)

    test_loss, test_acc = 0, 0

    with torch.inference_mode():  # Disable gradient calculations for efficiency
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)  # Move data to the device

            # 1. Forward pass: Compute model predictions
            test_pred_logits = model(X)

            # 2. Compute and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # 3. Compute accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)  # Get predicted class labels
            test_acc += (test_pred_labels == y).sum().item() / len(test_pred_labels)

    # Compute average loss and accuracy over all batches
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)

    return test_loss, test_acc

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          model_name: str,
          device: torch.device) -> Dict[str, List]:
    """
    Trains and evaluates the model for a given number of epochs.

    Args:
        model (torch.nn.Module): The neural network model.
        train_dataloader (torch.utils.data.DataLoader): DataLoader for training data.
        test_dataloader (torch.utils.data.DataLoader): DataLoader for testing data.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        loss_fn (torch.nn.Module): Loss function.
        epochs (int): Number of training epochs.
        model_name (str): Name of the model to save on directory.
        device (torch.device): Device to run the training (CPU/GPU).

    Returns:
        Dict[str, List]: Dictionary containing loss and accuracy for each epoch.
    """

    # Dictionary to store training and validation results
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    best_acc = 0.0
    # Loop over the specified number of epochs
    for epoch in tqdm(range(epochs), desc="Training Progress"):
        # Perform one training step
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )

        # Perform one test step
        test_loss, test_acc = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device
        )

        # Print results for current epoch
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            # Save the trained model
            utils.save_model(model=model,
                            target_dir="models",
                            model_name=f"{model_name}.pth")
            print(f"New best model saved at epoch {epoch+1} with test_acc={test_acc:.4f}")

        # Store results in the dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results  # Return the dictionary with loss/accuracy logs
