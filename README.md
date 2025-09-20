# HASTNet: A Hierarchical Adaptive Spatial Transformer Network for Cross-Model Generalazation in Medical Image Segmentation

## 📌Abstract

In medical image segmentation, cross-modal generalization of deep neural networks poses a key challenge. Existing methods are limited by modality-specific structural differences and boundary inconsistencies. To address this, we propose the Hierarchical Adaptive Spatial Transformer Network (HASTNet) for tackling modal differences and boundary deviations. Specifically, we first design Modality-Adaptive Attention Module (MAAM), which extracts hierarchical features through content-aware sparse attention, fusing modality-irrelevant multi-layer semantic information. Secondly, Hierarchical Receptive Field Aggregator (HRFA) is introduced to aggregate multi-scale receptive fields via long-range convolutions, capturing fine structures. Finally, to further correct boundary deviations, we propose Spatial Transformation Feature Correction Module (STFCM), which enhances the model's robustness to boundary regions by generating pseudo-variant features. Experimental results show that, compared to state-of-the-art methods, HASTNet achieves an average improvement of **2.50%** mIoU across nine datasets with only **9.9M** parameters, including a **5.25%** mIoU improvement on the CVC-ColonDB dataset.

## 🎇<font style="color:#000000;">Overall Architecture of HASTNet</font>

![](https://cdn.nlark.com/yuque/0/2025/jpeg/46378221/1758374638537-b7f079f9-3ed6-40ff-bffa-3d1c57708cfe.jpeg)

## 💡Key Features

+ We design MAAM using content-aware sparse attention to extract modality-irrelevant semantics, reducing the impact of medical imaging differences on model performance.
+ HRFA is introduced to aggregate features from receptive fields, capturing multi-scale structures in medical images.
+ We propose STFCM to dynamically correct boundary deviations by generating pseudo-modal variants via translation, and precisely refine segmentation boundaries.
+ Experiments on nine datasets across three modalities show HASTNet achieves the best segmentation performance, validating its superior robustness in cross-modal generalization.

## 📚Data Preparation

1. BUSI && Polyp Segmentation Dataset: [https://github.com/Xiaoqi-Zhao-DLUT/MSNet-M2SNet](https://github.com/Xiaoqi-Zhao-DLUT/MSNet-M2SNet)
2. DSB2018: [https://www.kaggle.com/c/data-science-bowl-2018](https://www.kaggle.com/c/data-science-bowl-2018)
3. STU: [https://drive.google.com/file/d/1k3OvEnYZaPWrng74aP4hAhgPXNHjpPj3/view?usp=drive_link](https://drive.google.com/file/d/1k3OvEnYZaPWrng74aP4hAhgPXNHjpPj3/view?usp=drive_link)
4. MonuSeg2018: [https://www.kaggle.com/datasets/tuanledinh/monuseg2018](https://www.kaggle.com/datasets/tuanledinh/monuseg2018)

The resulted file structure is as follows.

```plain
For train
dataset
├── Polyp_train
│   ├── train
│     ├── images
|     	├── ...
|     ├── targets
│     	├── ...
│   ├── val
│     ├── images
|     	├── ...
|     ├── targets
|     	├── ...
For test
dataset
├── Polyp_test
│   ├── CVC-300
│     ├── test
│     	├── images
|     		├── ...
|     	├── targets
│     		├── ...
│   ├── CVC-ClinicDB
│     ├── test
│     	├── images
|     		├── ...
|     	├── targets
│     		├── ...
│   ├── ...
```

## 🛠Code Usage

```plain
git clone https://github.com:Sengokuuuu/HASTNet.git
cd HASTNet
conda create -n HASTNet python=3.11
conda activate HASTNet
```

**We use PyTorch 2.5.1**

Training and testing configurations for all datasets are available in `train.sh` and `test.sh` respectively.  You can also run them directly.  

```plain
python main.py --method HASTNet --hiera_path "sam2_hiera_large.pt" --data_path "dataset/Polyp_Train" --work_dir "./work_dir/train_Polyp" --image_size 352 --label_size 352 --num_epochs 300 --batch_size 8 --num_cls 2 --eval_step 3 --lr 5e-4 --use_amp --detailed_metrics > Polyp.log 2>&1
```

```plain
python main.py --method HASTNet --hiera_path "sam2_hiera_large.pt" --data_path "./dataset/Polyp_Test/CVC-300" --label_size 352 --image_size 352 --num_cls 2 --use_amp --dist-url env://localhost:12345 --distributed --test_only --save_pic --resume "[model_weights]" 
```

## ⏳**Instructions**

**Full codebase and pre-trained weights for all datasets will be released upon paper acceptance. Coming soon~**

## 🎈**Acknowledgements**

**We sincerely thank everyone for their tremendous contributions to this project, and extend our gratitude to the editors and reviewers for their dedicated work!**

