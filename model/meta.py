import timm
import torch.nn as nn

class PoolFormer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = timm.create_model(args.CODER, pretrained=True,
                                    drop_path_rate=args.drop_path_rate,
                                    )
        
        num_head = self.encoder.head.fc.in_features
        self.encoder.head.fc = nn.Linear(num_head, 30)

    def forward(self, x):
        x = self.encoder(x)
        return x