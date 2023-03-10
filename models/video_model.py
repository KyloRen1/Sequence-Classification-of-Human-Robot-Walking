import torch
import torch.nn as nn
import torch.nn.functional as F

class StairNetVideoModel(nn.Module):
    def __init__(self, config, encoder: torch.nn.Module, encoder_output_channels: int):
        super().__init__()
        
        self.encoder = encoder 
        self.many_to_one = config.data_kwargs.many_to_one_setting
        self.temporal_model_name = config.model_kwargs.temporal.name
        
        if self.temporal_model_name == 'lstm':
            self.temporal = nn.LSTM(
                input_size = encoder_output_channels,
                hidden_size = config.model_kwargs.temporal.hidden_size,
                dropout = config.model_kwargs.temporal.dropout,
                num_layers = config.model_kwargs.temporal.n_layers,
                batch_first = False,
                bidirectional = False)
            self.decoder = nn.Linear(
                config.model_kwargs.temporal.hidden_size,
                config.model_kwargs.temporal.hidden_size,
            )
            self.decoder2 = nn.Linear(
                config.model_kwargs.temporal.hidden_size,
                config.data_kwargs.num_labels
            )
        else:
            raise NotImplementedError 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.temporal_model_name == 'lstm':

            output = list()
            hidden = None 

            for frame_idx in range(x.shape[1]):
                features = self.encoder(x[:, frame_idx, :, :, :])

                out, hidden = self.temporal(features, hidden)
                if not self.many_to_one:
                    idx_output = self.decoder2(F.relu(out))
                    output.append(idx_output)
            
            if not self.many_to_one:
                output = torch.stack(output)
                output = output.permute(1, 0, 2)
                return output
            else:
                output = self.decoder(out)
                output = F.relu(output)
                output = self.decoder2(output)
        else:
            raise NotImplementedError
        
        return output