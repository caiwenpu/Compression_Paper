---
title: Compressed Neural Network
date: 2017-10-30 18:16:32
tags:
mathjax: true
---
# Quantized Neural Network  
## low-precision Quantization 
### quantized weights only 
每个`float`权重近似表示成几个bits  


- <span style="color:red">Matthieu-Courbariaux,Yoshua-Bengio,Jean-Pierre-David:[BinaryConnect: Training Deep Neural Networks with binary weights during propagations. ][ref0] [NIPS 2015]</span>   
- <span style="color:red"> Jiaxiang Wu, Cong Leng, Yuhang Wang, Qinghao Hu, Jian Cheng:[Quantized Convolutional Neural Networks for Mobile Devices. ][ref24][CVPR 2016]</span>  
  * 量化卷积层的权重，调整全连接层的全精度权重；再量化全连接层的权重
  * 提供了实际加速的效果对比，3-4倍左右的加速
  * 提供了移动端的实验结果对比，包括内存开销，储存时间，储存开销
- <span style="color:red"> Chenzhuo Zhu, Song Han, Huizi Mao, William J. Dally:
  [Trained Ternary Quantization. ][ref7][ICLR 2017]</span> 
- <span style="color:red"> Aojun Zhou, Anbang Yao, Yiwen Guo, Lin Xu, Yurong Chen:
  [Incremental Network Quantization: Towards Lossless CNNs with Low-Precision Weights. ][ref4][ICLR 2017]</span> 
  * 简称INQ
  * 大的权重量化为2的幂次，小的权重保持全精度
  * 53$\times$压缩率，结合DNS
- <span style="color:red"> Yiwen Guo, Anbang Yao, Hao Zhao, Yurong Chen:[Network Sketching: Exploiting Binary Structure in Deep CNNs. ][ref22][CVPR 2017]</span> 
    * 剩余量化权重
    * 利用Bit平面之间的差值来减少冗余计算
    * 3bit 权重 Resnet18 1% top1精度损失 fine-tuning
- <span style="color:red">Felix Juefei-Xu, Vishnu Naresh Boddeti, Marios Savvides:
  [Local Binary Convolutional Neural Networks. ][ref23] [CVPR 2017] </span>
    * 一个三值卷积层(固定参数) + 一个1*1卷积层(可学习参数)替代一个完整卷积层
    * 更少的可训练参数，防止过拟合
- Learning Accurate Low-Bit Deep Neural Networks with Stochastic Quantization [BMVC 2017]
- <span style="color:red"> Cong Leng, Zesheng Dou, Hao Li, Shenghuo Zhu, Rong Jin:[Extremely Low Bit Neural Network: Squeeze the Last Bit Out with ADMM. ][ref24][AAAI 2018]</span>
   * 使用ADMM，优化损失函数，更新W
    * 只量化权重到 $\alpha_i \{0,1,-1,2,-2...2^N,-2^N\}$
    * 在Resnet VGG Googlenet 实验，效果略好于TTQ
- Training Quantized Nets: A Deeper Understanding. [NIPS 2017]
- Loss-Aware Weight Quantization of Deep Networks. [ICLR 2018]
- LEARNING DISCRETE WEIGHTS USING THE LOCAL REPARAMETERIZATION TRICK[ICLR 2018]
- From Hashing to CNNs: Training Binary Weight Networks via Hashing. [AAAI 2018]
- Learning Accurate Low-Bit Deep Neural Networks with Stochastic Quantization. [BMVC 2018]
- Explicit Loss-Error-Aware Quantization for Low-Bit Deep Neural Networks [CVPR 2018]
- An Optimal Control Approach to Deep Learning and Applications to Discrete-Weight Neural Networks. [ICML 2018]
- PROXQUANT: QUANTIZED NEURAL NETWORKS VIA PROXIMAL OPERATORS [ICLR 2019]


<!-- more -->
### quantized activations
- CATEGORICAL REPARAMETERIZATION WITH GUMBEL-SOFTMAX. [ICLR 2017]
- Towards Binary-Valued Gates for Robust LSTM Training. [ICML 2018]
- DEEP LEARNING AS A MIXED CONVEX- COMBINATORIAL OPTIMIZATION PROBLEM. [ICLR 2018]

### quantized weights and activations
- <span style="color:red">Itay Hubara, Matthieu Courbariaux, Daniel Soudry, Ran El-Yaniv, Yoshua Bengio:
  [Binarized Neural Networks. ][ref1] [NIPS 2016]</span>  
- <span style="color:red">Mohammad Rastegari, Vicente Ordonez, Joseph Redmon, Ali Farhadi:
  [XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks. ][ref2] [ECCV 2016]</span>  
    * 对于activation,每个spitial position都要一个scale，那么一个输入为$C \times H_{in} \times W_{in}$的feature map需要$H_{out} \times W_{out}$个scale。
    * 对于权重,一个3D的卷积核需要一个scale。
    * AlexNet 44.2% Top1 , 69.2% Top5
- <span style="color:red"> Zhaowei Cai, Xiaodong He, Jian Sun, Nuno Vasconcelos:
  [Deep Learning with Low Precision by Half-wave Gaussian Quantization. ][ref5][CVPR 2017]</span>  
- <span style="color:red">Shuchang Zhou, Zekun Ni, Xinyu Zhou, He Wen, Yuxin Wu, Yuheng Zou:
  [DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients. ][ref3][arXiv' 2016]</span> 
    * SVHN和Alexnet上实验，1bit权重，4bits输入，3%top1精度损失
- <span style="color:red">Zefan Li, Bingbing Ni, Wenjun Zhang, Xiaokang Yang, Wen Gao:
  [Performance Guaranteed Network Acceleration via High-Order Residual Quantization. ][ref17][ICCV 2017]</span>
    * 只在Mnist和Cifar10上做了实验
    * 一次卷积(一个卷积核)的浮点开销变为$\frac{K}{c_{in}k^2}$，K是量化阶数(一般取{1,2,3,4})
    * Cifar10的精度降低了3%,在一个自己构造的CNN
- <span style="color:red">Zhaowei Cai, Xiaodong He, Jian Sun, Nuno Vasconcelos:
  [Deep Learning with Low Precision by Half-Wave Gaussian Quantization. ][ref19] [CVPR 2017] </span>
    * 1bit权重，nbits输入
    * Alexnet,VGG,Resnet,GoogLenet
    * 1bit权重,2bits输入,Resnet18,10% top1损失,7%的top5损失;
- <span style="color:red">Wei Tang, Gang Hua, Liang Wang:[How to Train a Compact Binary Neural Network with High Accuracy? ][ref21][AAAI 2017]</span>
    * 整个激活一个scale，剩余量化激活，有scale。1bit 权重，无scale。
    * 使用PRelu作为激活函数，量化 -> 卷积 -> PReLU -> BN
    * 损失函数加入正则项 $\sum_i (1-w_i^2)$ 
- <span style="color:red">Wei Pan,Xiaofan Lin,Cong Zhao
  [Towards Accurate Binary Convolutional Neural Network. ][ref18][NIPS 2017]</span>
    * resnet18 上，5bits权重，5bits输入,4.3% top1精度损失,3.3% top5精度损失
    * 输入的scale在训练时固定，不用在测试时再去求
- Two-Step Quantization for Low-bit Neural Networks. [CVPR2018]
- WRPN: Wide Reduced-Precision Networks. [ICLR 2018]
- TRAINING AND INFERENCE WITH INTEGERS IN DEEP NEURAL NETWORKS. [ICLR 2018]
- Distribution-Aware Binarization of Neural Networks for Sketch Recognition [WACV 2018]
- Hybrid Binary Networks: Optimizing for Accuracy, Efficiency and Memory [WACV 2018]
- learning Compression from Limited Unlabeled Data [ECCV 2018]
- Bi-Real Net: Enhancing the Performance of 1-bit CNNs With Improved Representational Capability and Advanced Training Algorithm [ECCV 2018]
- Heterogeneous Bitwidth Binarization in Convolutional Neural Networks [NIPS 2018]
- Learning Low Precision Deep Neural Networks through Regularization [arXiv 2018]
- A SYSTEMATIC STUDY OF BINARY NEURAL NETWORKS’ OPTIMISATION [ICLR 2019]
- RELAXED QUANTIZATION FOR DISCRETIZED NEURAL NETWORKS [ICLR 2019]
- Composite Binary Decomposition Networks [AAAI 2019]
- Learning to Quantize Deep Networks by Optimizing Quantization Intervals with Task Loss [CVPR 2019]
- HAQ: Hardware-Aware Automated Quantization [CVPR 2019]
- A Main/Subsidiary Network Framework for Simplifying Binary Neural Networks [CVPR 2019]
- Binary Ensemble Neural Network: More Bits per Network or More Networks per Bit? [CVPR 2019]


[ref0]:http://papers.nips.cc/paper/5647-binaryconnect-training-deep-neural-networks-with-binary-weights-during-propagations
[ref1]: http://papers.nips.cc/paper/6573-binarized-neural-networks
[ref2]: https://arxiv.org/abs/1603.05279
[ref3]: https://arxiv.org/abs/1606.06160
[ref4]: https://arxiv.org/abs/1702.03044
[ref5]: https://arxiv.org/abs/1702.00953
[ref7]: https://arxiv.org/abs/1612.01064
[ref17]: https://arxiv.org/abs/1708.08687
[ref18]: http://papers.nips.cc/paper/6638-towards-accurate-binary-convolutional-neural-network
[ref19]: https://arxiv.org/abs/1702.00953
[ref21]: https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/download/14619/14454
[ref22]: https://arxiv.org/pdf/1706.02021.pdf
[ref23]: https://arxiv.org/abs/1608.06049
[ref24]: https://arxiv.org/abs/1512.06473v3



###Gradient Quantization

TernGrad: Ternary Gradients to Reduce Communication in Distributed Deep Learning [NIPS 2017]

Value-aware Quantization for Training and Inference of Neural Networks [ECCV 2018]



### Weight-Sharing Quantization    

一组权重共享同一个权重值
- <soan style="color:red"> Wenlin Chen, James T. Wilson, Stephen Tyree, Kilian Q. Weinberger, Yixin Chen:[Compressing Neural Networks with the Hashing Trick.][ref18] [ICML 2015] </span>
- <span style="color:red"> Song Han, Huizi Mao, William J. Dally:
  [Deep Compression: Compressing Deep Neural Network with Pruning, Trained Quantization and Huffman Coding][ref13][ICLR 2016 Best paper]</span>
- <span style="color:red"> Karen Ullrich, Edward Meeds, Max Welling:
  [Soft Weight-Sharing for Neural Network Compression.][ref12][ICLR 2017]</span>
- TOWARDS THE LIMIT OF NETWORK QUANTIZATION [ICLR 2017]
- Local Binary Convolutional Neural Networks. [CVPR 2017]
- <span style="color:red">Deep k-Means: Re-Training and Parameter Sharing with Harder Cluster
  Assignments for Compressing Deep Convolutions [ICML 2018]</span>
- <span style="color:red"> VARIATIONAL NETWORK QUANTIZATION [ICLR 2018]</span>
  - 优化变分下界，使用多个spike-and-slab组成的先验，使得最终权重区域几个离散的值
- Clustering Convolutional Kernels to Compress Deep Neural Networks [ECCV 2018]

[ref12]: https://arxiv.org/abs/1702.04008
[ref13]: https://arxiv.org/abs/1510.00149
[ref18]: http://proceedings.mlr.press/v37/chenc15.html

# Pruning Neural Network 
- <span style="color:red">Dong Yu,Frank Seide,Gang Li, Li Deng:[EXPLOITING SPARSENESS IN DEEP NEURAL NETWORKS FOR LARGE VOCABULARY
SPEECH RECOGNITION. ][ref25][ICASSP 2012]</span>
	 * 最早的剪枝？用于语音识别 
- <span style="color:red">Song Han, Jeff Pool, John Tran, William J. Dally:
[Learning both Weights and Connections for Efficient Neural Networks. ][ref6][NIPS 2015]</span>  
    * 在Alexnet上剪掉89%的参数，在VGG上剪掉92.5%的参数主要是全连接层
    * 在Lenet-5上保留8%的参数

- <span style="color:red">Wei Wen, Chunpeng Wu, Yandan Wang, Yiran Chen, Hai Li:
[Learning Structured Sparsity in Deep Neural Networks. ][ref8] [NIPS 2016] </span>   
    * 在Alexnet上5.1倍的CPU加速，3.1倍的GPU加速
- <span style="color:red">Yiwen Guo, Anbang Yao, Yurong Chen:
[Dynamic Network Surgery for Efficient DNNs. ][ref9] [NIPS 2016]</span>
    * 在Alexnet上17.7倍的压缩率
    * 在Lenet-5上只保留0.9%的参数
    * 比[NIPS 2015]的训练速度快7倍
- <span style="color:red"> Pavlo Molchanov, Stephen Tyree, Tero Karras, Timo Aila, Jan Kautz
NVIDIA:[PRUNING CONVOLUTIONAL NEURAL NETWORKS FOR RESOURCE EFFICIENT INFERENCE. ][ref16][ICLR 2017]</span>
- <span style="color:red">Jian-Hao Luo, Jianxin Wu, Weiyao Lin:
[ThiNet: A Filter Level Pruning Method for Deep Neural Network Compression. ][ref7][ICCV 2017]</span>  
    * 在VGG和Resnet-50上实验，数据集ImageNet，平台caffe
    * 可以在Resnet-50上剪掉50%的参数和FLOPS,1%精度损失
    * layer by layer 剪掉某些channel

[ref6]: https://arxiv.org/abs/1506.02626
[ref7]: https://arxiv.org/abs/1707.06342
[ref8]:http://papers.nips.cc/paper/6504-learning-structured-sparsity-in-deep-neural-networks
[ref9]:http://papers.nips.cc/paper/6165-dynamic-network-surgery-for-efficient-dnns
[ref16]: https://arxiv.org/abs/1611.06440v2

- Learning Efficient Convolutional Networks through Network Slimming. [ICCV 2017] 
- Variational Dropout Sparsifies Deep Neural Networks. [ICML 2017] 
- “Learning-Compression” Algorithms for Neural Net Pruning [CVPR 2018]
- PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning [CVPR 2018]
- Soft Filter Pruning for Accelerating Deep Convolutional Neural Networks [IJCAI 2018]
- A Systematic DNN Weight Pruning Framework using Alternating Direction Method of Multipliers [ECCV 2018]
- Data-Driven Sparse Structure Selection for Deep Neural Network [ECCV 2018]
- Discrimination-aware Channel Pruning for Deep Neural Networks [NIPS 2018]
- Dynamic Channel Pruning: Feature Boosting and Suppression [ICLR 2019]
- RePr: Improved Training of Convolutional Filters [CVPR 2019]

# Matrix Decomposition

- <span style="color:red"> Emily L. Denton, Wojciech Zaremba, Joan Bruna, Yann LeCun, Rob Fergus:
[Exploiting Linear Structure Within Convolutional Networks for Efficient Evaluation. ][ref10][NIPS 2014]</span>    
    * 2.5倍加速 在Alexnet上 2-3倍卷积层参数减少 ，小于1%的精度损失
- <span style="color:red"> Vadim Lebedev, Yaroslav Ganin, Maksim Rakhuba, Ivan V. Oseledets, Victor S. Lempitsky:
[Speeding-up Convolutional Neural Networks Using Fine-tuned CP-Decomposition. ][ref11][ICLR 2015] </span>  
    * 4.5倍加速 在Alexnet conv2上
    * $1 \times 1$ conv $\rightarrow$ 一个feature map的垂直conv $\rightarrow$ 一个feature map的水平conv-> $1\times 1$ conv
- <span style="color:red">Cheng Tai, Tong Xiao, Yi Zhang, Xiaogang Wang, Weinan E:[CONVOLUTIONAL NEURAL NETWORKS WITH LOW- RANK REGULARIZATION. ][ref20][ICLR 2016] </span>
    * 扩展Jaderberg et al.(2014)
    * 一个卷积层分解成两个1D卷积层，不适用矩阵分解的最优解调整，从头训练Alexnet得到更好的效果(需要BN)
    * VGG16 2.75倍的压缩率
- On Compressing Deep Models by Low Rank and Sparse Decomposition. [CVPR 2017]
- <span style="color:red">Jose M. Alvarez,Mathieu Salzmann:[Compression-aware Training of Deep Networks][NIPS 2017]</span>
    * Resnet50 27% 参数压缩
    * 在Decomposed网络上从头开始训练，Decompose:把一个2D卷积层分成两个1D卷积层，中间加入一个激活函数


[ref10]: https://arxiv.org/abs/1404.0736
[ref11]: https://arxiv.org/abs/1412.6553
[ref20]: https://arxiv.org/abs/1511.06067
[ref25]: http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6288897

#Hardware

- FINN: A Framework for Fast, Scalable Binarized Neural Network Inference. [FPGA 2017]