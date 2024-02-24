# TalkNet+DeiT

>[**IMPROVING AUDIOVISUAL ACTIVE SPEAKER DETECTION IN EGOCENTRIC RECORDINGS WITH THE DATA-EFFICIENT IMAGE TRANSFORMER**]
> ASRU 2023

![image](https://github.com/jclarke98/TalkNet-DeiT/blob/main/TalkNet%2BDeiT_1.png)

This work extends the [TalkNet-ASD](https://github.com/TaoRuijie/TalkNet_ASD/blob/main/utils/tools.py#L34) architecture with the [Data Efficient image Transformer](https://arxiv.org/abs/2012.12877) to inject contextual information provided by full-scene images to help compensate for the noise present in egocentric recordings.

## Data preprocessing

Preprocessing scripts have been slightly modified from the [Ego4D-AVD benchmark implementation of TalkNet](https://github.com/zcxu-eric/Ego4d_TalkNet_ASD).

### From Scratch
To start from scratch without any prior processed annotation, you must have the Ego4D file structure, outlined [here](https://github.com/zcxu-eric/Ego4d_TalkNet_ASD), as follows:


data/
* json/
  * av_train.json
  * av_val.json
  * av_test_unannotated.json
* split/
  * test.list
  * train.list
  * val.list
  * full.list
* track_results/
  * 00407bd8-37b4-421b-9c41-58bb8f141716.txt
  * 007beb60-cbab-4c9e-ace4-f5f1ba73fccf.txt
  * ...

```
python utils/annot_preprocess.py --basePath {path to Ego4d_TalkNet_ASD} --split train
python utils/annot_preprocess.py --basePath {path to Ego4d_TalkNet_ASD} --split val
python utils/annot_preprocess.py --basePath {path to Ego4d_TalkNet_ASD} --split test
```

### With Existing Processed Annotation

If annotation already exists for the csv files and bounding box jsons, ensure they are structued as per the standard Ego4d_TalkNet_ASD folder:

Ego4d_TalkNet_ASD/
* data/
    * ego4d/
        * csv/
            * active_speaker_train.csv
            * active_speaker_val.csv
        * bbox/
            * ...
    * infer/
        * csv/
            * ...
        * bbox/
            * ...

### Tensor Preprocessing

The following code can be used to preprocess the tensors in Ego4D to significantly reduce training time. The code preprocesses the visual components of Ego4D-AVD into trackwise pytorch tensors for TalkNet+DeiT and should be ran in parallel across multiple HPC sessions simultaneously.

Training & validation folds: 
```
python tensor_grabber.py --annotPath {path to ego4d/csv & bbox} --split {train/val} --dataPath {path to video_imgs} --savePath {path to save tensors}
```
Evaluation fold:
```
python tensor_grabber.py --annotPath {path to infer/csv & bbox} --split val --dataPath {path to video_imgs} --savePath {path to save tensors}
```

Run the following across a single session to fill any missing tracks.

Training & validation folds:
```
python tensor_grabber.py --annotPath {path to ego4d/csv & bbox} --split {train/val} --dataPath {path to video_imgs} --savePath {path to save tensors} --fillPass
```
Evaluation fold:
```
python tensor_grabber.py --annotPath {path to infer/csv & bbox} --split val --dataPath {path to video_imgs} --savePath {path to save tensors} --fillPass
```

## Model Training

```
python trainTalkNetDeiT.py
```

## Inference

```
python inferTalkNetDeiT.py
```

## Citation
Please cite:
```
@INPROCEEDINGS{10389764,
  author={Clarke, Jason and Gotoh, Yoshihiko and Goetze, Stefan},
  booktitle={2023 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU)}, 
  title={Improving Audiovisual Active Speaker Detection in Egocentric Recordings with the Data-Efficient Image Transformer}, 
  year={2023},
  volume={},
  number={},
  pages={1-8},
  keywords={Conferences;Benchmark testing;Transformers;Feature extraction;Data models;Acoustics;Recording;Active speaker detection;context modelling;data-efficient image transformers},
  doi={10.1109/ASRU57964.2023.10389764}}
```

This work builds upon existing work so please cite the following as well: 
```
@inproceedings{tao2021someone,
  title={Is Someone Speaking? Exploring Long-term Temporal Features for Audio-visual Active Speaker Detection},
  author={Tao, Ruijie and Pan, Zexu and Das, Rohan Kumar and Qian, Xinyuan and Shou, Mike Zheng and Li, Haizhou},
  booktitle = {Proceedings of the 29th ACM International Conference on Multimedia},
  pages = {3927–3935},
  year={2021}
}
```