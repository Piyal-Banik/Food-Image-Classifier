import torch
from pathlib import Path

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """
    Saves the state dictionary of a PyTorch model to a specified directory.

    Args:
        model (torch.nn.Module): The PyTorch model to save.
        target_dir (str): The directory where the model should be saved.
        model_name (str): The name of the saved model file (must end with '.pt' or '.pth').

    Raises:
        AssertionError: If the model_name does not end with '.pt' or '.pth'.

    Example:
        save_model(model=my_model, target_dir="models", model_name="best_model.pth")
    """

    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                            exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
                f=model_save_path)
