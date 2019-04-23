---
title: Compressed Neural Network
date: 2017-10-30 18:16:32
tags:
mathjax: true
---
# Quantized Neural Network  
## low-precision Quantization 
### quantized weights only 
每个`float`权重近似表示成几个bits  


- <span style="color:red">Matthieu-Courbariaux,Yoshua-Bengio,Jean-Pierre-David:["BinaryConnect: Training Deep Neural Networks with binary weights during propagations."][ref0] [NIPS 2015]</span>   

- <span style="color:red"> Jiaxiang Wu, Cong Leng, Yuhang Wang, Qinghao Hu, Jian Cheng:[Quantized Convolutional Neural Networks for Mobile Devices][ref24][CVPR 2016]</span>  
  * 量化卷积层的权重，调整全连接层的全精度权重；再量化全连接层的权重
  * 提供了实际加速的效果对比，3-4倍左右的加速
  * 提供了移动端的实验结果对比，包括内存开销，储存时间，储存开销

- <span style="color:red"> Chenzhuo Zhu, Song Han, Huizi Mao, William J. Dally:
  [Trained Ternary Quantization][ref7][ICLR 2017]</span> 

- <span style="color:red"> Aojun Zhou, Anbang Yao, Yiwen Guo, Lin Xu, Yurong Chen:
  ["Incremental Network Quantization: Towards Lossless CNNs with Low-Precision Weights."][ref4][ICLR 2017]</span> 
  * 简称INQ
  * 大的权重量化为2的幂次，小的权重保持全精度
  * 53$\times$压缩率，结合DNS

- <span style="color:red"> Yiwen Guo, Anbang Yao, Hao Zhao, Yurong Chen:[Network Sketching: Exploiting Binary Structure in Deep CNNs][ref22][CVPR 2017]</span> 
    * 剩余量化权重
    * 利用Bit平面之间的差值来减少冗余计算
    * 3bit 权重 Resnet18 1% top1精度损失 fine-tuning

- <span style="color:red">Felix Juefei-Xu, Vishnu Naresh Boddeti, Marios Savvides:
  [Local Binary Convolutional Neural Networks.][ref23] [CVPR 2017] </span>
    * 一个三值卷积层(固定参数) + 一个1*1卷积层(科学系参数)替代一个完整卷积层
    * 更少的可训练参数，防止过拟合

- Training Quantized Nets: A Deeper Understanding [NIPS 2017]

- Learning Accurate Low-Bit Deep Neural Networks with Stochastic Quantization [BMVC 2017]

- <span style="color:red">LEARNING DISCRETE WEIGHTS USING THE LOCAL
   REPARAMETERIZATION TRICK[ICLR 2018]</span>

   - 每个权重是一个随机变量，由一组参数刻画。
   - 由中心极限定理，这一层的输出服从正太分布

- <span style="color:red">An Optimal Control Approach to Deep Learning and
   Applications to Discrete-Weight Neural Networks[ICML 2018]</span>

   - 理论依据的离散权重神经网络的训练。

- <span style="color:red">Deep Neural Network Compression with Single and Multiple Level Quantization[AAAI 2018]</span>

      - 增量聚类量化，对权重进行聚类，基于每一个类簇的聚类(量化)对损失函数的影响，把所有的类簇划分量化部分和非量化部分，量化损失函数大的量化部分，重新训练剩余的非量化部分
      - 不再每层单独考虑，所有层一起考虑，对所有层的聚类量化损失在一起比较。
      - 效果略好于INQ，不是很明显。

- <span style="color:red"> Cong Leng, Zesheng Dou, Hao Li, Shenghuo Zhu, Rong Jin:[Extremely Low Bit Neural Network: Squeeze the Last Bit Out with ADMM][ref24][AAAI 2018]</span>

   - 使用ADMM，优化损失函数，更新W
   - 只量化权重到 $\alpha_i \{0,1,-1,2,-2...2^N,-2^N\}$
   - 在Resnet VGG Googlenet 实验，效果略好于TTQ

- <span style="color:red">LOSS-AWARE WEIGHT QUANTIZATION OF DEEP NET-
   WORKS [ICLR 2018]</span>

   - 基于近端牛顿法，对带有离散权重约束的目标进行优化
   - 优化过程中依然保留浮点权重，权重更新在浮点权重上进行

- <span style="color:red">PROXQUANT: QUANTIZED NEURAL NETWORKS VIA
   PROXIMAL OPERATORS [ICLR 2019]</span>

- LEARNING RECURRENT BINARY/TERNARY WEIGHTS [ICLR 2019] 


   - Add BatchNorm to LSTM for Quantized weights

- Projection Convolutional Neural Networks for 1-bit CNNs via Discrete Back Propagation [AAAI 2019]

- Simultaneously Optimizing Weight and Quantizer of Ternary Neural Network using Truncated Gaussian Approximation [CVPR 2019]

- 


   - 

<!-- more -->

### quantized weights and activations
- <span style="color:red">Itay Hubara, Matthieu Courbariaux, Daniel Soudry, Ran El-Yaniv, Yoshua Bengio:
  ["Binarized Neural Networks."][ref1] [NIPS 2016]</span>  
- <span style="color:red">Mohammad Rastegari, Vicente Ordonez, Joseph Redmon, Ali Farhadi:
  ["XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks."][ref2] [ECCV 2016]</span>  
    * 对于activation,每个spitial position都要一个scale，那么一个输入为$C \times H_{in} \times W_{in}$的feature map需要$H_{out} \times W_{out}$个scale。
    * 对于权重,一个3D的卷积核需要一个scale。
    * AlexNet 44.2% Top1 , 69.2% Top5
- <span style="color:red"> Zhaowei Cai, Xiaodong He, Jian Sun, Nuno Vasconcelos:
  ["Deep Learning with Low Precision by Half-wave Gaussian Quantization."][ref5][CVPR 2017]</span>  
- <span style="color:red">Shuchang Zhou, Zekun Ni, Xinyu Zhou, He Wen, Yuxin Wu, Yuheng Zou:
  ["DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients"][ref3][arXiv' 2016]</span> 

    * SVHN和Alexnet上实验，1bit权重，4bits输入，3%top1精度损失
- <span style="color:red">Zefan Li, Bingbing Ni, Wenjun Zhang, Xiaokang Yang, Wen Gao:
  [Performance Guaranteed Network Acceleration via High-Order Residual Quantization.][ref17][ICCV 2017]</span>
    * 只在Mnist和Cifar10上做了实验
    * 一次卷积(一个卷积核)的浮点开销变为$\frac{K}{c_{in}k^2}$，K是量化阶数(一般取{1,2,3,4})
    * Cifar10的精度降低了3%,在一个自己构造的CNN
- <span style="color:red">Zhaowei Cai, Xiaodong He, Jian Sun, Nuno Vasconcelos:
  [Deep Learning with Low Precision by Half-Wave Gaussian Quantization.][ref19] [CVPR 2017] </span>
    * 1bit权重，nbits输入
    * Alexnet,VGG,Resnet,GoogLenet
    * 1bit权重,2bits输入,Resnet18,10% top1损失,7%的top5损失;
- <span style="color:red">Wei Tang, Gang Hua, Liang Wang:[How to Train a Compact Binary Neural Network with High Accuracy?][ref21][AAAI 2017]</span>
    * 整个激活一个scale，剩余量化激活，有scale。1bit 权重，无scale。
    * 使用PRelu作为激活函数，量化 -> 卷积 -> PReLU -> BN
    * 损失函数加入正则项 $\sum_i (1-w_i^2)$ 
- <span style="color:red">Wei Pan,Xiaofan Lin,Cong Zhao
  [Towards Accurate Binary Convolutional Neural Network][ref18][NIPS 2017]</span>
  - resnet18 上，5bits权重，5bits输入,4.3% top1 精度损失,3.3% top5精度损失
  - 输入的scale在训练时固定，不用在测试时再去求
- <span style="color:red">ALTERNATING MULTI-BIT QUANTIZATION FOR
  RECURRENT NEURAL NETWORKS [ICLR 2018]</span>
  - 缩放因子+比特平面，重构全精度的权重/激活，最小化重构误差，迭代求解缩放因子、比特平面
  - 针对RNN
- SYQ: Learning Symmetric Quantization For Efficient Deep Neural Networks [CVPR 2018]
- Towards Effective Low-bitwidth Convolutional Neural Networks [CVPR 2018]
  - Train BNN for 32bit to 16bit to 8bit to 4bit to 2bit
- <span style="color:red">Two-Step Quantization for Low-bit Neural Networks[CVPR 2018]</span>
  - 权重量化和激活量化分开做，先训练量化激活的网络，求一个好的离散权重的初始解，训练量化激活和权重的网络
- Learning Low Precision Deep Neural Networks through Regularization [arXiv 2018]
    - Add Quantization Regularization item 
- Bi-Real Net: Enhancing the Performance of 1-bit CNNs With Improved Representational Capability and Advanced Training Algorithm [ECCV 2018]
    - add a extra shortcut in between two conv in one res-block.
- TBN: Convolutional Neural Network with Ternary Inputs and Binary Weights [ECCV 2018]
- <span style="color:red">LQ-Nets: Learned Quantization for Highly
    Accurate and Compact Deep Neural Networks [ECCV 2018]</span>
    - 最小化每一层的量化重构误差
- <span style="color:red">Towards Binary-Valued Gates for Robust LSTM Training [ICML 2018]</span>
    - Gumbel-Softmax 训练趋近二值的LSTM Gate
- Heterogeneous Bitwidth Binarization in Convolutional Neural Networks [NIPS 2018]
- HitNet: Hybrid Ternary Recurrent Neural Network [NIPS 2018]
- <span style="color:red">RELAXED QUANTIZATION FOR
    DISCRETIZED NEURAL NETWORKS [ICLR 2019]</span>
    - Gumbel-Softmax STE 训练离散的激活函数
- DEFENSIVE QUANTIZATION: WHEN EFFICIENCY MEETS ROBUSTNESS [ICLR 2019]
    - leverage Quantized activation to boost defense
- A SYSTEMATIC STUDY OF BINARY NEURAL NETWORKS’ OPTIMISATION [ICLR 2019]
    - How to Train BNN. Hyper-Parameter settings
- A Main/Subsidiary Network Framework for Simplifying Binary Neural Networks [CVPR 2019]
    - Prune BNN with mask trained 
- Training Quantized Network with Auxiliary Gradient Module [arXiv 2019]

    - Likely Knowledge Distill / Attention Transfer. Add Auxiliary Loss to every layer
- HAQ: Hardware-Aware Automated Quantization [CVPR 2019]
    - Reinforce Learning to allocate different bits for different layers
- Structured Binary Neural Networks for Accurate Image Classification and Semantic Segmentation [CVPR 2019]
- Regularizing Activation Distribution for Training Binarized Deep Networks [CVPR 2019]
    - Regularizing Batch-Normlize output to the scope of [-1,1]
- Learning to Quantize Deep Networks by Optimizing Quantization Intervals with Task Loss [CVPR 2019]
    - Learn Quantized funtions intervals
- Matrix and tensor decompositions for training binary neural networks [arXiv 2019]
    - Add capacity of float weight to get more accurate binary weight



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



### Gradient Quantization

TernGrad: Ternary Gradients to Reduce Communication in Distributed Deep Learning [NIPS 2017]

Value-aware Quantization for Training and Inference of Neural Networks [ECCV 2018]



# Weight-Sharing Quantization  

一组权重共享同一个权重值
- <soan style="color:red"> Wenlin Chen, James T. Wilson, Stephen Tyree, Kilian Q. Weinberger, Yixin Chen:[Compressing Neural Networks with the Hashing Trick.][ref18] [ICML 2015] </span>
- <span style="color:red"> Song Han, Huizi Mao, William J. Dally:
  [Deep Compression: Compressing Deep Neural Network with Pruning, Trained Quantization and Huffman Coding][ref13][ICLR 2016 Best paper]</span>
- <span style="color:red"> Karen Ullrich, Edward Meeds, Max Welling:
  [Soft Weight-Sharing for Neural Network Compression.][ref12][ICLR 2017]</span>
- TOWARDS THE LIMIT OF NETWORK QUANTIZATION [ICLR 2017]
- <span style="color:red">Deep k-Means: Re-Training and Parameter Sharing with Harder Cluster
  Assignments for Compressing Deep Convolutions [ICML 2018]</span>
- <span style="color:red"> VARIATIONAL NETWORK QUANTIZATION [ICLR 2018]</span>
  - 优化变分下界，使用多个spike-and-slab组成的先验，使得最终权重区域几个离散的值
- Clustering Convolutional Kernels to Compress Deep Neural Networks [ECCV 2018]
- Coreset-Based Neural Network Compression [ECCV 2018]
- Learning Versatile Filters for Efficient Convolutional Neural Networks [NIPS 2018]
- Exploiting Kernel Sparsity and Entropy for Interpretable CNN Compression [CVPR 2019]

[ref12]: https://arxiv.org/abs/1702.04008
[ref13]: https://arxiv.org/abs/1510.00149
[ref18]: http://proceedings.mlr.press/v37/chenc15.html

# Pruning Neural Network 
- <span style="color:red">Dong Yu,Frank Seide,Gang Li, Li Deng:[EXPLOITING SPARSENESS IN DEEP NEURAL NETWORKS FOR LARGE VOCABULARY
  SPEECH RECOGNITION][ref25][ICASSP 2012]</span>
   * 最早的剪枝？用于语音识别 
- <span style="color:red">Song Han, Jeff Pool, John Tran, William J. Dally:
  [Learning both Weights and Connections for Efficient Neural Networks.][ref6][NIPS 2015]</span>  
    * 在Alexnet上剪掉89%的参数，在VGG上剪掉92.5%的参数主要是全连接层
    * 在Lenet-5上保留8%的参数
- <span style="color:red">Wei Wen, Chunpeng Wu, Yandan Wang, Yiran Chen, Hai Li:
  [Learning Structured Sparsity in Deep Neural Networks.][ref8] [NIPS 2016] </span>   
    * 在Alexnet上5.1倍的CPU加速，3.1倍的GPU加速
- <span style="color:red">Yiwen Guo, Anbang Yao, Yurong Chen:
  [Dynamic Network Surgery for Efficient DNNs.][ref9] [NIPS 2016]
  </span>
    * 在Alexnet上17.7倍的压缩率
    * 在Lenet-5上只保留0.9%的参数
    * 比[NIPS 2015]的训练速度快7倍
- Variational Dropout Sparsifies Deep Neural Networks [ICML 2017]
- <span style="color:red"> Pavlo Molchanov, Stephen Tyree, Tero Karras, Timo Aila, Jan Kautz
  NVIDIA:[PRUNING CONVOLUTIONAL NEURAL NETWORKS FOR RESOURCE EFFICIENT INFERENCE][ref16][ICLR 2017]</span>
- <span style="color:red">Jian-Hao Luo, Jianxin Wu, Weiyao Lin:
  [ThiNet: A Filter Level Pruning Method for Deep Neural Network Compression.][ref7][ICCV 2017]</span>  
  - 在VGG和Resnet-50上实验，数据集ImageNet，平台caffe
  - 可以在Resnet-50上剪掉50%的参数和FLOPS,1%精度损失
  - layer by layer 剪掉某些channel
- Learning Efficient Convolutional Networks through Network Slimming [ICCV 2017]
  - Reg BN scale with L1-norm to prune channel
- Structured Bayesian Pruning via Log-Normal Multiplicative Noise [NIPS 2017]
- Learning to Prune Deep Neural Networks via Layer-wise Optimal Brain Surgeon [NIPS 2017]
- Recovering from Random Pruning: On the Plasticity of Deep Convolutional Neural Networks [WACV 2018]
- RETHINKING THE SMALLER-NORM-LESS-INFORMATIVE ASSUMPTION IN CHANNEL PRUNING OF CONVOLUTION LAYERS [ICLR 2018]
- LEARNING TO SHARE: SIMULTANEOUS PARAMETER TYING AND SPARSIFICATION IN DEEP LEARNING [ICLR 2018]
- LEARNING SPARSE NEURAL NETWORKS THROUGH L0 REGULARIZATION [ICLR 2018]
- “Learning-Compression” Algorithms for Neural Net Pruning [CVPR 2018]
- PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning [CVPR 2018]
- Soft Filter Pruning for Accelerating Deep Convolutional Neural Networks [IJCAI 2018]
- A Systematic DNN Weight Pruning Framework using Alternating Direction Method of Multipliers [ECCV 2018]
- Data-Driven Sparse Structure Selection for Deep Neural Network [ECCV 2018]
- Discrimination-aware Channel Pruning for Deep Neural Networks [NIPS 2018]
- Synaptic Strength For Convolutional Neural Network [NIPS 2018]
- Learning Sparse Neural Networks via Sensitivity-Driven Regularization [NIPS 2018]
- Frequency-Domain Dynamic Pruning for Convolutional Neural Networks [NIPS 2018]
- Dynamic Channel Pruning: Feature Boosting and Suppression [ICLR 2019]
- SNIP: SINGLE-SHOT NETWORK PRUNING BASED ON CONNECTION SENSITIVITY [ICLR 2019]
- ENERGY-CONSTRAINED COMPRESSION FOR DEEP NEURAL NETWORKS VIA WEIGHTED SPARSE PROJEC-
  TION AND LAYER INPUT MASKING [ICLR 2019]
- RePr: Improved Training of Convolutional Filters [CVPR 2019]
- Pruning Filter via Geometric Median for Deep Convolutional Neural Networks Acceleration [CVPR 2019]
- Centripetal SGD for Pruning Very Deep Convolutional Networks with Complicated Structure [CVPR 2019]
- Fully Learnable Group Convolution for Acceleration of Deep Neural Networks [CVPR 2019]
- On Implicit Filter Level Sparsity in Convolutional Neural Networks [CVPR 2019]
- ECC: Platform-Independent Energy-Constrained Deep Neural Network Compression via a Bilinear Regression Model [CVPR 2019]
- Towards Optimal Structured CNN Pruning via Generative Adversarial Learning [CVPR 2019]
- Cascaded Projection: End-to-End Network Compression and Acceleration [CVPR 2019]

[ref6]: https://arxiv.org/abs/1506.02626
[ref7]: https://arxiv.org/abs/1707.06342
[ref8]:http://papers.nips.cc/paper/6504-learning-structured-sparsity-in-deep-neural-networks
[ref9]:http://papers.nips.cc/paper/6165-dynamic-network-surgery-for-efficient-dnns
[ref16]: https://arxiv.org/abs/1611.06440v2



# Matrix Decomposition

- <span style="color:red"> Emily L. Denton, Wojciech Zaremba, Joan Bruna, Yann LeCun, Rob Fergus:
  [Exploiting Linear Structure Within Convolutional Networks for Efficient Evaluation.][ref10][NIPS 2014]</span>    
    * 2.5倍加速 在Alexnet上 2-3倍卷积层参数减少 ，小于1%的精度损失
- <span style="color:red"> Vadim Lebedev, Yaroslav Ganin, Maksim Rakhuba, Ivan V. Oseledets, Victor S. Lempitsky:
  [Speeding-up Convolutional Neural Networks Using Fine-tuned CP-Decomposition.][ref11][ICLR 2015] </span>  
    * 4.5倍加速 在Alexnet conv2上
    * $1 \times 1$ conv $\rightarrow$ 一个feature map的垂直conv $\rightarrow$ 一个feature map的水平conv-> $1\times 1$ conv
- <span style="color:red">Cheng Tai, Tong Xiao, Yi Zhang, Xiaogang Wang, Weinan E:[CONVOLUTIONAL NEURAL NETWORKS WITH LOW- RANK REGULARIZATION][ref20][ICLR 2016] </span>
    * 扩展Jaderberg et al.(2014)
    * 一个卷积层分解成两个1D卷积层，不适用矩阵分解的最优解调整，从头训练Alexnet得到更好的效果(需要BN)
    * VGG16 2.75倍的压缩率
- Accelerating Very Deep Convolutional Networks for Classification and Detection [TPAMI 2016]
- Tensor-Train Recurrent Neural Networks for Video Classification [ICML 2017]
- Domain-adaptive deep network compression [ICCV 2017]
- Coordinating Filters for Faster Deep Neural Networks [ICCV 2017]
- <span style="color:red">Jose M. Alvarez,Mathieu Salzmann:[Compression-aware Training of Deep Networks </span> [NIPS 2017]
    - Resnet50 27% 参数压缩
    - 在Decomposed网络上从头开始训练，Decompose:把一个2D卷积层分成两个1D卷积层，中间加入一个激活函数
- Learning Compact Recurrent Neural Networks with Block-Term Tensor Decomposition [CVPR 2018]
- Wide Compression: Tensor Ring Nets [CVPR 2018]
- Extreme Network Compression via Filter Group Approximation [ECCV 2018]
- Trained Rank Pruning for Efficient Deep Neural Networks [arXiv 2018]
- Compressing Recurrent Neural Networks with Tensor Ring for Action Recognition [AAAI 2019]
- T-Net: Parametrizing Fully Convolutional Nets with a Single High-Order Tensor [CVPR 2019]
- 


[ref10]: https://arxiv.org/abs/1404.0736
[ref11]: https://arxiv.org/abs/1412.6553
[ref20]: https://arxiv.org/abs/1511.06067
[ref25]: http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6288897



# Knowledge Distill

- Distilling the Knowledge in a Neural Network [2014]
- FITNETS: HINTS FOR THIN DEEP NETS [ICLR 2015]
- PAYING MORE ATTENTION TO ATTENTION:IMPROVING THE PERFORMANCE OF CONVOLUTIONAL
  NEURAL NETWORKS VIA ATTENTION TRANSFER [ICLR 2017]
- Mimicking Very Efficient Network for Object Detection [CVPR 2017]
- A Gift from Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning [CVPR 2017]
- Deep Mutual Learning [CVPR 2018]
- Data Distillation: Towards Omni-Supervised Learning [CVPR 2018]
- Quantization Mimic: Towards Very Tiny CNNfor Object Detection [ECCV 2018]
- KDGAN: Knowledge Distillation with Generative Adversarial Networks [NIPS 2018]
- Knowledge Distillation by On-the-Fly Native Ensemble [NIPS 2018]
- Paraphrasing Complex Network: Network Compression via Factor Transfer [NIPS 2018]
- Knowledge Transfer via Distillation of Activation Boundaries Formed by Hidden Neurons [AAAI 2019]
- Relational Knowledge Distillation [CVPR 2019]
- Snapshot Distillation: Teacher-Student Optimization in One Generation [CVPR 2019]