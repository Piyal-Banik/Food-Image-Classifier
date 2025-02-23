import torch
import torch.nn as nn
import torchvision.models as models

class FoodImageClassifier(nn.Module):
    """
    A deep learning model based on EfficientNet-B0 for food image classification.
    
    Attributes:
        model (torchvision.models.EfficientNet): Pretrained EfficientNet-B0 model.
    """

    def __init__(self, fine_tune: bool, num_classes: int):
        """
        Initializes the EfficientNet-B0 model and modifies it for food classification.
        
        Args:
            fine_tune (bool): If True, allows training of EfficientNet’s feature extractor.
                              If False, freezes feature extractor layers.
            num_classes (int): Number of output classes for classification.
        """
        super().__init__()

        # Load EfficientNet-B0 with pretrained weights
        weights = models.EfficientNet_B0_Weights.DEFAULT
        self.model = models.efficientnet_b0(weights=weights)

        # Fine-tune logic
        if not fine_tune:
            print("[INFO]: Freezing feature extractor layers...")
            for param in self.model.features.parameters():
                param.requires_grad = False  # Freezes entire feature extractor

        # Modify classifier head to match the number of classes
        in_features = self.model.classifier[1].in_features  # Get input size of original classifier
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=in_features, out_features=num_classes, bias=True)  # Custom output layer
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            X (torch.Tensor): Input batch of images (B, C, H, W).
        
        Returns:
            torch.Tensor: Predicted class logits.
        """
        return self.model(X)
