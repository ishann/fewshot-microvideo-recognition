import torch

class ss_slowfast_r50(torch.nn.Module):

    def __init__(self):
        super(ss_slowfast_r50, self).__init__()
        self.encoder = torch.hub.load('facebookresearch/pytorchvideo',
                                      'slowfast_r50', pretrained=True) 
        self.activation = torch.nn.ReLU()
        self.linear = torch.nn.Linear(400, 2)

    def forward(self, x):

        kinetics_embedding = self.encoder(x)
        kinetics_embedding = self.activation(kinetics_embedding)
        ss_embedding = self.linear(kinetics_embedding)
        
        return ss_embedding

