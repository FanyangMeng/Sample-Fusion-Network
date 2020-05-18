# Sample Fusion Network
Sample Fusion Network: An End-to-End Data Augmentation Network for Skeleton-based Human Action Recognition
https://ieeexplore.ieee.org/document/8704987


The processed data of NTU-RGB+D can be obtained by Baidu Cloud:
Link：https://pan.baidu.com/s/1hdrkaiAlctPdIX7hwgKmFg
Extraction Code ：8jbp

For a given skeleton data x in the NTU RGB+D dataset, the data preprocessing is as follows：

1), the size of x is reshaped to N×150.

For a single-person action (N×25×3), x will be copied and concat to N×25×3×2, and then reshaped to N×150; For a two-person mutual action (N×25×3×2), x will be reshaped to N×150 directly.

2), resize N×150 to 100×150
