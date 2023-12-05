# WireframeParsers

We collect and rewrite the code for wireframe parsing of different deep-learning models.

## Available datasets

| Dataset      | Paper | Source | Resource |
|--------------|-------|--------|----------|
| YorkUrban    | [Efficient Edge-Based Methods for Estimating Manhattan Frames in Urban Imagery](https://elderlab.yorku.ca/wp-content/uploads/2017/02/pdenisThesis.pdf) | ECCV 2008 | [project](https://www.elderlab.yorku.ca/resources/york-urban-line-segment-database-information/) |
| YorkUrban<sup>1</sup> | [A Novel Linelet-Based Representation for Line Segment Detection](https://ieeexplore.ieee.org/document/7926451) | TPAMI 2018 | [repo](https://github.com/NamgyuCho/Linelet-code-and-YorkUrban-LineSegment-DB/tree/1dae1378153e5ec2d15de68489d7e4f6ca865e5e)  |
| ShanghaiTech | [Learning to Parse Wireframes in Images of Man-Made Environments](https://arxiv.org/abs/2007.07527v1) | CVPR 2018 | [repo](https://github.com/huangkuns/wireframe/tree/d76e7406b3581ca8df26da0e0a4ec3bd14a8184d) |

<sup>1</sup> This is another version of annotation to original images.

Before run the scripts we provide, you need to rebuild annotations according to COCO api:
```bash
python datasets/build.py <dataset> <root>
```

## Available models

| Model  | Paper | Source | Resource |
|--------|-------|--------|----------|
| L-CNN  | [End-to-End Wireframe Parsing](https://arxiv.org/abs/1905.03246v3) | ICCV 2019 | [repo](https://github.com/zhou13/lcnn/tree/57524636bc4614a32beac1af3b31f66ded2122ae) |
| HAWPv1 | [Holistically-Attracted Wireframe Parsing](https://arxiv.org/abs/2003.01663v1) | CVPR 2020  | [repo](https://github.com/cherubicXN/hawp/tree/21391181150e05654a0ac26ba3ea226a7fd725cc) |
| TP-LSD | [TP-LSD: Tri-Points Based Line Segment Detector](https://arxiv.org/abs/2009.05505v1) | ECCV 2020 | [repo](https://github.com/Siyuada7/TP-LSD/tree/a00cb36712e5c7adbdcad1a9d319dde6b53472c6) |
| LETR   | [Line Segment Detection Using Transformers without Edges](https://arxiv.org/abs/2101.01909v2) | CVPR 2021 | [repo](https://github.com/mlpc-ucsd/LETR/tree/6022fbd9df65569f4a82b1ac065bee8f26fc4ca6) |
| F-Clip | [Fully Convolutional Line Parsing](https://arxiv.org/abs/2104.11207v3) | Neurocomputing 2022 | [repo](https://github.com/Delay-Xili/F-Clip/tree/e30d307e728aa530b5601e4581510bcd6093b620) |
| HAWPv2 | [Holistically-Attracted Wireframe Parsing: From Supervised Learning to Self-Supervised Learning](https://arxiv.org/abs/2210.12971v2) | TPAMI 2023 | [repo](https://github.com/cherubicXN/hawp/tree/027c39753933da8713e579130976616f380ce54d) |

## Setup

Download model weights through the links below:

| Model      | Size | Link |
|------------|-----:|------|
| L-CNN      | 150M | https://drive.google.com/file/d/1NvZkEqWNUBAfuhFPNGiCItjy4iU0UOy2 |
| HAWPv1     | 40M  | https://github.com/cherubicXN/hawp-torchhub/releases/download/0.1/model-hawp-hg-5d31f70.pth |
| TP-LSD-HG  | 29M  | https://github.com/Siyuada7/TP-LSD/blob/a00cb36712e5c7adbdcad1a9d319dde6b53472c6/pretraineds/HG128.pth |
| TP-LSD-Res | 92M  | https://github.com/Siyuada7/TP-LSD/blob/a00cb36712e5c7adbdcad1a9d319dde6b53472c6/pretraineds/Res512.pth |
| LETR-R101  | 437M | https://vcl.ucsd.edu/letr/checkpoints/res101/res101_stage2_focal.zip |
| LETR-R50   | 364M | https://vcl.ucsd.edu/letr/checkpoints/res50/res50_stage2_focal.zip |
| F-Clip     | 122M | https://drive.google.com/file/d/1jMZB-kYTNAGaVPW0FBLTAQMc3m3vR-_P |
| HAWPv2     | 43M  | https://github.com/cherubicXN/hawp-torchhub/releases/download/HAWPv2/hawpv2-edb9b23f.pth |

All model weights should be put under folder `./weights` (for zip files, you should unzip them first and only put weight files here).

## Usage

Draw the detected lines of the first image in ShanghaiTech dataset (test):
```bash
python demo.py <model>
```

Save the detection results:
```bash
python inference.py <dataset> <model>
```

Evaluate strutrual average presision:
```bash
python eval-sAP.py <dataset> <model>
```

## Evaluation Results

ShanghaiTech Dataset:

| Model         | sAP5 | sAP10 | sAP15 | FPS   |
|---------------|-----:|------:|------:|------:|
| L-CNN         | 59.3 | 63.4  | 65.2  | 9.47  |
| HAWPv1        | 62.7 | 66.7  | 68.5  | 18.58 |
| LETR-R101     | 59.1 | 65.1  | 67.6  | 3.68  |
| LETR-R50      | 58.1 | 64.2  | 66.7  | 4.14  |
| F-Clip-HG2    | 61.9 | 66.6  | 68.7  | 12.10 |
| F-Clip-HG2-LB | 63.3 | 67.6  | 69.4  | 10.30 |
| HAWPv2        | 64.3 | 68.5  | 70.3  | 18.18 |

Yorkurban Dataset:

| Model         | sAP5 | sAP10 | sAP15 | FPS   |
|---------------|-----:|------:|------:|------:|
| L-CNN         | 26.3 | 28.7  | 30.0  | 3.59  |
| HAWPv1        | 27.4 | 30.0  | 31.4  | 11.40 |
| LETR-R101     | 23.8 | 27.7  | 29.7  | 3.61  |
| LETR-R50      | 25.1 | 29.3  | 31.5  | 4.22  |
| F-Clip-HG2    | 26.7 | 29.0  | 20.6  | 10.60 |
| F-Clip-HG2-LB | 27.3 | 29.7  | 31.2  | 9.71  |
| HAWPv2        | 27.7 | 30.3  | 31.9  | 13.02 |

Note: LETR is run on images resized to 1100.