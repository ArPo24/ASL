import torch
import torch.nn as nn
import torch.nn.functional as F

class ASLModel(nn.Module):
    def __init__(self, num_classes=29):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=2)
        self.conv2 =nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Автоматический расчет размера для полносвязного слоя
        self._to_linear = None
        with torch.no_grad():  
            test_input = torch.randn(1, 3, 200, 200)
            test_output = self._get_conv_output(test_input)
            self._to_linear = test_output[0].numel() 
        
        # Полносвязные слои
        self.fc1 = nn.Linear(self._to_linear, 512)  
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def _get_conv_output(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x
        
    def forward(self, x):
        x = self._get_conv_output(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x