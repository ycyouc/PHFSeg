# Lung CT image segmentation based on spectrum Transformer and Triplet Attention
In the field of automatic segmentation of lung CT images, deep learning-based methods are maturing and have achieved remarkable results, and the accuracy of segmentation has been continuously improved. But due to the lack of unified large-scale datasets with annotations, the quality of annotations varies among different datasets. Unbalanced categories of segmented images, blurred boundaries between infected and healthy regions, and different sizes of lesions, etc., lead to the fact that the existing methods still cannot fully meet the requirements of segmentation accuracy in medical applications. In this paper, a lung CT image segmentation method based on spectrum Transformer and triplet attention is proposed. The method fuses the computed spectrum Transformer module and the triplet attention module in a parallel way by means of the Parallel Hybrid Fusion Module (PHFM), which extracts global and local contextual information from two perspectives: sequence features and cross-dimensional features. We propose the Spectrum Transformer Block(STB). The module utilizes Fast Fourier Transform to learn the weights of each frequency component in the spectral space. In addition, we perform extensive experimental validation and comparative analysis on two publicly available datasets to demonstrate its generalizability, and conduct a comparative study with state-of-the-art models. The results show that our model PHFSeg has advantages over other competing state-of-the-art (SOTA) methods for CT image segmentation tasks.



## Training and Testing
We use Python 3.9, PyTorch 1.13.1 to test the code. The `train.py` script is for training, and the `test.py` script is for testing.

Before running the code, you should first put the images, masks, and data lists into the `datasets` folder. The file structure is as follows:
```
.
├── PHFSeg
│   ├── module
│   └── datasets
│   └── results
│   └── PHFSeg.py
│   └── train.py
│   └── test.py
│   └── ...
```

For convenience, we provide our data on [Google Drive](https://drive.google.com/file/d/1EScMnIZTOwpPROC7dl6Z3q-83zJm5pwb/view?usp=sharing). Please download it and unzip it into the `$ROOT_DIR` directory.

### Training PHFSeg

```
python train.py --max_epochs 100 --batch_size 16 --lr 1e-3 --lr_mode poly --savedir ./results_PHFSeg_crossVal --model_name PHFSeg --data_name P20
```

### Testing PHFSeg

For example, we use the following command to test PHFSeg on the COVID-19-P20 dataset:
```
python test.py --model_name PHFSeg --data_name P20 --pretrained Pretrained/COVID-19-P20/<MODEL_NAME> --savedir ./outputs
```
The generated segmentation maps of five folds will be outputted into the folder of `$ROOT_DIR/outputs/P20/PHFSeg/crossVal0~crossVal4`, respectively.

