import os
import math
import timm
import torch
from .movinet import StairNetMoViNet
from movinets.config import _C
from .video_model import StairNetVideoModel


def create_encoder(config):
    if config.model_kwargs.encoder.name in timm.list_models('mobile*', pretrained=True):
        model = timm.create_model(
            config.model_kwargs.encoder.name, 
            pretrained=config.model_kwargs.encoder.pretrained, 
            num_classes=0
        )
        if 'mobilenet' in config.model_kwargs.encoder.name:
            if config.model_kwargs.encoder.freeze is not None:
                for param in model.blocks[:config.model_kwargs.encoder.freeze].parameters():
                    param.requires_grad = False

            encoder_output = model.feature_info[-1]['num_chs'] * 4

        elif 'mobilevit' in config.model_kwargs.encoder.name:
            if config.model_kwargs.encoder.freeze is not None:
                for param in model.stages[:config.model_kwargs.encoder.freeze].parameters():
                    param.requires_grad = False
            encoder_output = model.feature_info[-1]['num_chs']
    else:
        raise NotImplementedError
    return model, encoder_output

def create_model(config):
    if config.model_kwargs.encoder.name == 'movinet':
        model = StairNetMoViNet(
            _C.MODEL.MoViNetA2,
            many_to_one = config.data_kwargs.many_to_one_setting,
            seq_len = config.data_kwargs.seq_len,
            causal = False,
            pretrained = True,
            num_classes = config.data_kwargs.num_labels)
    else:
        encoder, encoder_output = create_encoder(config)
        model = StairNetVideoModel(config, encoder, encoder_output)
    return model