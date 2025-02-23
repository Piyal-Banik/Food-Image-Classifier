import torch
import argparse
import data_setup, training_pipeline, model, utils
from torchvision import transforms

# Argument parser for command-line execution
parser = argparse.ArgumentParser(description="Train a deep learning model on a custom dataset.")

# Add arguments for hyperparameters and settings
parser.add_argument("--model_name", type=str, default="model", help="Name of the model for saving.")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs.")
parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer.")
parser.add_argument("--train_dir", type=str, required=True, help="Path to the training dataset.")
parser.add_argument("--test_dir", type=str, required=True, help="Path to the test dataset.")
parser.add_argument("--img_size", type=int, default=224, help="Image size for resizing (assumes square images).")
parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device to train on.")

# Parse arguments
args = parser.parse_args()

# Setup target device
device = torch.device("cuda" if torch.cuda.is_available() and args.device == "auto" else args.device)

# Create transforms for image preprocessing
data_transform = transforms.Compose([
    transforms.Resize((args.img_size, args.img_size)),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    )
])

# Load datasets using data_setup module
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=args.train_dir,
    test_dir=args.test_dir,
    transform=data_transform,
    batch_size=args.batch_size
)

# Dynamically create model based on dataset class count
food_model = model.FoodImageClassifier( 
    fine_tune=True, num_classes=len(class_names) 
).to(device)

# Define loss function and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(food_model.parameters(), lr=args.learning_rate)

# Train the model
training_pipeline.train(model=food_model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=args.num_epochs,
             model_name=args.model_name,
             device=device)
