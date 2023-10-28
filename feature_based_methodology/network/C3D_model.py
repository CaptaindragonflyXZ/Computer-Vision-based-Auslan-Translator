import torch
import torch.nn as nn
from mypath import Path

frame_num = 40


class Linear3DModel(nn.Module):
    def __init__(self, num_classes=10, pretrained=None):
        super(Linear3DModel, self).__init__()

        self.input_dim = frame_num * 3 * 4 * 21  # Linearized input dimensions

        # Fully connected layers
        self.fc1 = nn.Linear(self.input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_classes)

        # Weight initialization
        self.__init_weight()

        # If a pretrained path is provided, load the weights
        if pretrained:
            self.__load_pretrained_weights(pretrained)

    def forward(self, x):
        # print ('ori:',x.size())
        x = x.reshape(-1, self.input_dim) 
        # print ('0:',x.size())
        x = nn.ReLU()(self.fc1(x))
        # print ('1:',x.size())
        x = nn.ReLU()(self.fc2(x))
        # print ('2:',x.size())
        x = nn.ReLU()(self.fc3(x))
        # print ('3:',x.size())
        x = self.fc4(x)
        # print ('4:',x.size())
        return x

    def __init_weight(self):
        """Private method for weight initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def __load_pretrained_weights(self, path):
        """Private method to load pretrained weights into the model."""
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)

    
def get_1x_lr_params(model):
    """
    This generator returns all the parameters for conv and two fc layers of the net.
    """
    b = [model.fc1, model.fc2, model.fc3]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k

def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last fc layer of the net.
    """
    b = [model.fc4]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k

# Usage example
# model = Linear3DModel(pretrained_path="path_to_pretrained_weights.pth")



# =======================================================================================================================================
if __name__ == "__main__":
    # inputs = torch.rand(1, 3, 16, 112, 112)
    inputs = torch.rand(1, 3, 50, 4, 21)
    print('size is', inputs.shape)
    net = Linear3DModel(num_classes=4, pretrained=False)

    outputs = net.forward(inputs)
    print(outputs.size())