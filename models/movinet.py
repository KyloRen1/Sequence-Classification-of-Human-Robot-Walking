import torch
import torch.nn as nn

from movinets import MoViNet
from movinets.models import BasicBneck, ConvBlock3D, TemporalCGAvgPool3D, Swish

from collections import OrderedDict


class ClassificationHeads(torch.nn.Module):
    def __init__(self, in_shape, out_shape, seq_len):
        super().__init__()

        self.seq_len = seq_len
        for i in range(self.seq_len):
            net = torch.nn.Linear(in_shape, out_shape)
            setattr(self, f'head_{i}', net)

    def forward(self, x):
        x = x.squeeze()
        output = list()

        for i in range(self.seq_len):
            net = getattr(self, f'head_{i}')
            out = net(x)
            output.append(out)
        output = torch.stack(output)
        print(output.shape)
        output = output.permute(1, 0, 2)
        print(output.shape)
        return output
    

class Reshape(torch.nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        x = x.squeeze()
        BS, _ = x.shape
        x = torch.reshape(x, (BS, *self.shape))
        return x
    

class StairNetMoViNet(MoViNet):
  def __init__(self,
                 cfg,
                 many_to_one,
                 seq_len: int = 5,
                 causal: bool = True,
                 pretrained: bool = False,
                 num_classes: int = 600,
                 conv_type: str = "3d",
                 tf_like: bool = False
                 ) -> None:
        super().__init__(cfg)
        """
        causal: causal mode
        pretrained: pretrained models
        If pretrained is True:
            num_classes is set to 600,
            conv_type is set to "3d" if causal is False,
                "2plus1d" if causal is True
            tf_like is set to True
        num_classes: number of classes for classifcation
        conv_type: type of convolution either 3d or 2plus1d
        tf_like: tf_like behaviour, basically same padding for convolutions
        """
        if pretrained:
            tf_like = True
            conv_type = "2plus1d" if causal else "3d"
        blocks_dic = OrderedDict()

        norm_layer = nn.BatchNorm3d if conv_type == "3d" else nn.BatchNorm2d
        activation_layer = Swish if conv_type == "3d" else nn.Hardswish

        # conv1
        self.conv1 = ConvBlock3D(
            in_planes=cfg.conv1.input_channels,
            out_planes=cfg.conv1.out_channels,
            kernel_size=cfg.conv1.kernel_size,
            stride=cfg.conv1.stride,
            padding=cfg.conv1.padding,
            causal=causal,
            conv_type=conv_type,
            tf_like=tf_like,
            norm_layer=norm_layer,
            activation_layer=activation_layer
            )
        # blocks
        for i, block in enumerate(cfg.blocks):
            for j, basicblock in enumerate(block):
                blocks_dic[f"b{i}_l{j}"] = BasicBneck(basicblock,
                                                      causal=causal,
                                                      conv_type=conv_type,
                                                      tf_like=tf_like,
                                                      norm_layer=norm_layer,
                                                      activation_layer=activation_layer
                                                      )
        self.blocks = nn.Sequential(blocks_dic)
        # conv7
        self.conv7 = ConvBlock3D(
            in_planes=cfg.conv7.input_channels,
            out_planes=cfg.conv7.out_channels,
            kernel_size=cfg.conv7.kernel_size,
            stride=cfg.conv7.stride,
            padding=cfg.conv7.padding,
            causal=causal,
            conv_type=conv_type,
            tf_like=tf_like,
            norm_layer=norm_layer,
            activation_layer=activation_layer
            )
        # pool
        self.classifier = nn.Sequential(
            # dense9
            ConvBlock3D(cfg.conv7.out_channels,
                        cfg.dense9.hidden_dim,
                        kernel_size=(1, 1, 1),
                        tf_like=tf_like,
                        causal=causal,
                        conv_type=conv_type,
                        bias=True),
            Swish(),
            nn.Dropout(p=0.2, inplace=True),
        )
        if many_to_one:
            self.classifier_head = nn.Sequential(
                nn.Conv3d(
                    cfg.dense9.hidden_dim, 
                    num_classes, 
                    kernel_size=(1,1,1)),
                nn.Flatten(start_dim=1)
            )
        else:
            self.classifier_head = nn.Sequential(
                nn.Conv3d(
                        in_channels = cfg.dense9.hidden_dim, 
                        out_channels = num_classes * seq_len, 
                        kernel_size = (1,1,1)), 
                ClassificationHeads(
                        in_shape = num_classes * seq_len,
                        out_shape = num_classes,
                        seq_len = seq_len) 
            )

        if causal:
            self.cgap = TemporalCGAvgPool3D()
        if pretrained:
            if causal:
                if cfg.name not in ["A0", "A1", "A2"]:
                    raise ValueError("Only A0,A1,A2 streaming" +
                                     "networks are available pretrained")
                pretrained_dict = (torch.hub
                              .load_state_dict_from_url(cfg.stream_weights))
            else:
                pretrained_dict = torch.hub.load_state_dict_from_url(cfg.weights)
            
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict) 
            self.load_state_dict(model_dict)
        else:
            self.apply(self._weight_init)
        self.causal = causal

  def _forward_impl(self, x):
    x = self.conv1(x)
    x = self.blocks(x)
    x = self.conv7(x)
    x = self.avg(x)
    x = self.classifier(x)
    x = self.classifier_head(x)
    return x