# Sequential Image Classification of Human-Robot Walking Environments using Temporal Neural Networks

[Sequential Image Classification of Human-Robot Walking Environments using Temporal Neural Networks]() by Bohdan Ivaniuk-Skulskyi, Andrew Garrett Kurbis, Alex Mihailidis3,4, and Brokoslaw Laschowski


## Installation
```python
python3 -m venv venv
source venv/bin/activate
pip install git+https://github.com/Atze00/MoViNet-pytorch.git
pip install -r requirements.txt
```

## Model Zoo

<table>
  <tr>
    <th>Name</th>
    <th>Parameters</th>
    <th>GFLOPs</th>
    <th>Resolution</th>
    <th>Accuracy</th>
    <th>F1-score</th>
    <th colspan="2">Download</th>
  </tr>
<tr>
    <td>MoViNet</td>
    <td>4.03M</td>
    <td>2.5</td>
    <td>5x3x224x224</td>
    <td>0.983</td>
    <td>0.982</td>
    <td><a href="https://github.com/KyloRen1/Sequence-Classification-of-Human-Robot-Walking/releases/download/0.0.1/movinet_m2o.pth">model</a></td>
    <td><a href="https://github.com/KyloRen1/Sequence-Classification-of-Human-Robot-Walking/blob/main/configs/movinet_m2o.yaml">config</td>
  </tr>
  <tr>
    <td>MobileViT-LSTM</td>
    <td>3.36M</td>
    <td>9.84</td>
    <td>5x3x224x224</td>
    <td>0.970</td>
    <td>0.968</td>
    <td><a href="https://github.com/KyloRen1/Sequence-Classification-of-Human-Robot-Walking/releases/download/0.0.1/mobilevit_xxs_lstm_m2o.pth">model</a></td>
    <td><a href="https://github.com/KyloRen1/Sequence-Classification-of-Human-Robot-Walking/blob/main/configs/mobilevit_lstm_m2o.yaml">config</a></td>
  </tr>
  <tr>
    <td>MobileNet-LSTM</td>
    <td>6.08M</td>
    <td>53.96</td>
    <td>5x3x224x224</td>
    <td>0.973</td>
    <td>0.970</td>
    <td><a href="https://github.com/KyloRen1/Sequence-Classification-of-Human-Robot-Walking/releases/download/0.0.1/mobilenetv2_100_lstm_m2o.pth">model</a></td>
    <td><a href="https://github.com/KyloRen1/Sequence-Classification-of-Human-Robot-Walking/blob/main/configs/mobilenet_lstm_m2o.yaml">config</a></td>
  </tr>
  <tr>
    <td>MobileNet-LSTM (seq2seq)</td>
    <td>5.93M</td>
    <td>50.97</td>
    <td>5x3x224x224</td>
    <td>0.707</td>
    <td>0.799</td>
    <td><a href="https://github.com/KyloRen1/Sequence-Classification-of-Human-Robot-Walking/releases/download/0.0.1/mobilenetv2_100_lstm_m2m.pth">model</a></td>
    <td><a href="https://github.com/KyloRen1/Sequence-Classification-of-Human-Robot-Walking/blob/main/configs/mobilenet_lstm_m2m.yaml">config</a></td>
  </tr>
  <tr>
    <td><a href="https://ieeexplore.ieee.org/document/9896501">Baseline (Kurbis et al.)</a></td>
    <td>2.26M</td>
    <td>0.61</td>
    <td>3x224x224</td>
    <td>0.972</td>
    <td>0.972</td>
    <td>-</td>
    <td>-</td>
  </tr>
</table>


## Inference
Download one checkpoint with its configuration file and run the following command
```python
python test.py --experiment_cfg CONFIG.yaml \
  --dataset_folder $DATASET \
  --val_samples_file /Users/bogdanivanyuk/Desktop/data_stairnet/video_records/validation.txt --test_samples_file /Users/bogdanivanyuk/Desktop/data_stairnet/video_records/test.txt
```

## Train 
```python
python train.py --experiment_cfg CONFIG.yaml \
  --dataset_folder $DATASET \
  --val_samples_file /Users/bogdanivanyuk/Desktop/data_stairnet/video_records/validation.txt --test_samples_file /Users/bogdanivanyuk/Desktop/data_stairnet/video_records/test.txt
```

## Confusion matrices

## BibTex
```
```

## Acknowledgements
The visual encoder models are taken from [timm](https://github.com/huggingface/pytorch-image-models) library and the MoViNet is based on [this](https://github.com/Atze00/MoViNet-pytorch) implementation