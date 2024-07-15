# Title: Lung CT image segmentation based on spectrum Transformer and Triplet Attention
In the field of automatic segmentation of lung CT images, deep learning-based methods are maturing and have achieved remarkable results, and the accuracy of segmentation has been continuously improved. But due to the lack of unified large-scale datasets with annotations, the quality of annotations varies among different datasets. Unbalanced categories of segmented images, blurred boundaries between infected and healthy regions, and different sizes of lesions, etc., lead to the fact that the existing methods still cannot fully meet the requirements of segmentation accuracy in medical applications. In this paper, a lung CT image segmentation method based on spectrum Transformer and triplet attention is proposed. The method fuses the computed spectrum Transformer module and the triplet attention module in a parallel way by means of the Parallel Hybrid Fusion Module (PHFM), which extracts global and local contextual information from two perspectives: sequence features and cross-dimensional features. We propose the Spectrum Transformer Block(STB). The module utilizes Fast Fourier Transform (FFT) to learn the weights of each frequency component in the spectral space. In addition, we perform extensive experimental validation and comparative analysis on two publicly available datasets to demonstrate its generalizability, and conduct a comparative study with state-of-the-art models. The results show that our model PHFSeg has advantages over other competing state-of-the-art (SOTA) methods for CT image segmentation tasks.


## Preparation

## Training and Testing
We use Python 3.9, PyTorch 1.13.1 to test the code. The `train.py` script is for training, and the `test.py` script is for testing.

