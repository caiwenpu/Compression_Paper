---
title: Compressed Neural Network
date: 2017-10-30 18:16:32
tags:
mathjax: true
---
# Quantized Neural Network  
## low-precision Quantization 
### quantized weights only 
replace float-precision weights by low-precision or n-bit weights


- <span style="color:red">Matthieu-Courbariaux,Yoshua-Bengio,Jean-Pierre-David:["BinaryConnect: Training Deep Neural Networks with binary weights during propagations."][ref0] [NIPS 2015]</span>   

- <span style="color:red"> Jiaxiang Wu, Cong Leng, Yuhang Wang, Qinghao Hu, Jian Cheng:[Quantized Convolutional Neural Networks for Mobile Devices][ref24][CVPR 2016]</span>  
  * quantize conv weights -> fine-tune fc weights -> quantize fc weights
  * mobile device runtime and memory size.

- <span style="color:red"> Chenzhuo Zhu, Song Han, Huizi Mao, William J. Dally:
  [Trained Ternary Quantization][ref7][ICLR 2017]</span> 

- <span style="color:red"> Aojun Zhou, Anbang Yao, Yiwen Guo, Lin Xu, Yurong Chen:
  ["Incremental Network Quantization: Towards Lossless CNNs with Low-Precision Weights."][ref4][ICLR 2017]</span> 
  * quantize part of weights, fine-tune rest float weight util all weights are quatized.
  * 53$\times$ compression rate in companion with DNS.

- <span style="color:red"> Yiwen Guo, Anbang Yao, Hao Zhao, Yurong Chen:[Network Sketching: Exploiting Binary Structure in Deep CNNs][ref22][CVPR 2017]</span> 
  * Residual Quantization and Residual Quantization with refinement
  
- <span style="color:red">Felix Juefei-Xu, Vishnu Naresh Boddeti, Marios Savvides:
  [Local Binary Convolutional Neural Networks.][ref23] [CVPR 2017] </span>
    * shared Ternary weights tensor with filter-wise scale.
    * less trainable parameters to deal overfitting.

- Training Quantized Nets: A Deeper Understanding [NIPS 2017]

- Learning Accurate Low-Bit Deep Neural Networks with Stochastic Quantization [BMVC 2017]

- <span style="color:red">LEARNING DISCRETE WEIGHTS USING THE LOCAL
   REPARAMETERIZATION TRICK[ICLR 2018]</span>

     * Varitional method

- <span style="color:red">An Optimal Control Approach to Deep Learning and
   Applications to Discrete-Weight Neural Networks[ICML 2018]</span>

     * Control-Theory deal with Discrete-Weight Optimization。

- Deep Neural Network Compression with Single and Multiple Level Quantization[AAAI 2018]
      * loss-aware weights group partition to low-precison and full-precison.
      * Layer seperable quantization to Global quantization。

- **Cong Leng, Zesheng Dou, Hao Li, Shenghuo Zhu, Rong Jin:[Extremely Low Bit Neural Network: Squeeze the Last Bit Out with ADMM][ref24][AAAI 2018]**
      * ADMM,  constrain add to loss func. 

- <span style="color:red">LOSS-AWARE WEIGHT QUANTIZATION OF DEEP NET-
   WORKS [ICLR 2018]</span>
      * projected-newton method
      * keep float weights in training

- <span style="color:red">PROXQUANT: QUANTIZED NEURAL NETWORKS VIA
   PROXIMAL OPERATORS [ICLR 2019]</span>

- LEARNING RECURRENT BINARY/TERNARY WEIGHTS [ICLR 2019] 
      * Add BatchNorm to LSTM for Quantized weights

- Projection Convolutional Neural Networks for 1-bit CNNs via Discrete Back Propagation [AAAI 2019]

- Simultaneously Optimizing Weight and Quantizer of Ternary Neural Network using Truncated Gaussian Approximation [CVPR 2019]



<!-- more -->

### quantized weights and activations
- <span style="color:red">Itay Hubara, Matthieu Courbariaux, Daniel Soudry, Ran El-Yaniv, Yoshua Bengio:
  ["Binarized Neural Networks."][ref1] [NIPS 2016]</span>  
- <span style="color:red">Mohammad Rastegari, Vicente Ordonez, Joseph Redmon, Ali Farhadi:
  ["XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks."][ref2] [ECCV 2016]</span>  
    * Activation: Slice-Wise Scale。
    * Weight: Filter-Wise Scale。
    * AlexNet 44.2% Top1 , 69.2% Top5
- <span style="color:red"> Zhaowei Cai, Xiaodong He, Jian Sun, Nuno Vasconcelos:
  ["Deep Learning with Low Precision by Half-wave Gaussian Quantization."][ref5][CVPR 2017]</span>  
- <span style="color:red">Shuchang Zhou, Zekun Ni, Xinyu Zhou, He Wen, Yuxin Wu, Yuheng Zou:
  ["DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients"][ref3][arXiv' 2016]</span> 
    * tanh + min-max transforms weights to [0,1], then quantize
    * clip [0,1] trainsform activation to [0,1], then quantize
- <span style="color:red">Zefan Li, Bingbing Ni, Wenjun Zhang, Xiaokang Yang, Wen Gao:
  [Performance Guaranteed Network Acceleration via High-Order Residual Quantization.][ref17][ICCV 2017]</span>
    * Residual quantize activation.
- <span style="color:red">Zhaowei Cai, Xiaodong He, Jian Sun, Nuno Vasconcelos:
  [Deep Learning with Low Precision by Half-Wave Gaussian Quantization.][ref19] [CVPR 2017] </span>
    * approx activation to (0,1)-Gaussian, then get quantize values by this std gaussian distribution.
- <span style="color:red">Wei Tang, Gang Hua, Liang Wang:[How to Train a Compact Binary Neural Network with High Accuracy?][ref21][AAAI 2017]</span>
    * Residual quantize activation
    * PReLU replace Relu
    
- <span style="color:red">Wei Pan,Xiaofan Lin,Cong Zhao
  [Towards Accurate Binary Convolutional Neural Network][ref18][NIPS 2017]</span>
- <span style="color:red">ALTERNATING MULTI-BIT QUANTIZATION FOR
  RECURRENT NEURAL NETWORKS [ICLR 2018]</span>
    * Alternative Quantizetion to minimize quantization error.
    * RNN only
- SYQ: Learning Symmetric Quantization For Efficient Deep Neural Networks [CVPR 2018]
- Towards Effective Low-bitwidth Convolutional Neural Networks [CVPR 2018]
  - Train BNN for 32bit to 16bit to 8bit to 4bit to 2bit
- <span style="color:red">Two-Step Quantization for Low-bit Neural Networks[CVPR 2018]</span>
  - quantization activation -> get init quantized weights -> fine-tune quantized weights
- Learning Low Precision Deep Neural Networks through Regularization [arXiv 2018]
    - Add Quantization Regularization item 
- Bi-Real Net: Enhancing the Performance of 1-bit CNNs With Improved Representational Capability and Advanced Training Algorithm [ECCV 2018]
    - add a extra shortcut in between two conv in one res-block.
- TBN: Convolutional Neural Network with Ternary Inputs and Binary Weights [ECCV 2018]
- <span style="color:red">LQ-Nets: Learned Quantization for Highly
    Accurate and Compact Deep Neural Networks [ECCV 2018]</span>
    - Alternative Quantizetion to minimize quantization error.
    - CNN.
- <span style="color:red">Towards Binary-Valued Gates for Robust LSTM Training [ICML 2018]</span>
    - Gumbel-Softmax to get binarized LSTM Gate
- Heterogeneous Bitwidth Binarization in Convolutional Neural Networks [NIPS 2018]
- HitNet: Hybrid Ternary Recurrent Neural Network [NIPS 2018]
- <span style="color:red">RELAXED QUANTIZATION FOR
    DISCRETIZED NEURAL NETWORKS [ICLR 2019]</span>
    - Gumbel-Softmax STE to trained discrete weights and activation
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
- **Structured Binary Neural Networks for Accurate Image Classification and Semantic Segmentation** [CVPR 2019]
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



### Gradient Quantization && Distributed Training

- TernGrad: Ternary Gradients to Reduce Communication in Distributed Deep Learning [NIPS 2017]
- QSGD: Communication-Efficient SGD via Gradient Quantization and Encoding [NIPS 2017]
- Value-aware Quantization for Training and Inference of Neural Networks [ECCV 2018]


### Quantize weights && activation && gradients Simultaneously

- Flexpoint: An Adaptive Numerical Format for Efficient Training of Deep Neural Networks [NIPS 2017]
- TRAINING AND INFERENCE WITH INTEGERS IN DEEP NEURAL NETWORKS [ICLR 2018]
- Training Deep Neural Networks with 8-bit Floating Point Numbers [NIPS 2018]
- Training DNNs with Hybrid Block Floating Point [NIPS 2018]


# Weight-Sharing Quantization  

A group of weights sharing one value 
- <soan style="color:red"> Wenlin Chen, James T. Wilson, Stephen Tyree, Kilian Q. Weinberger, Yixin Chen:[Compressing Neural Networks with the Hashing Trick.][ref18] [ICML 2015] </span>
- Compressing Convolutional Neural Networks in the Frequency Domain [KDD 2016]
    * DCT transform to Frequency Domain. 
- <span style="color:red"> Song Han, Huizi Mao, William J. Dally:
  [Deep Compression: Compressing Deep Neural Network with Pruning, Trained Quantization and Huffman Coding][ref13][ICLR 2016 Best paper]</span>
- <span style="color:red"> Karen Ullrich, Edward Meeds, Max Welling:
  [Soft Weight-Sharing for Neural Network Compression.][ref12][ICLR 2017]</span>
- TOWARDS THE LIMIT OF NETWORK QUANTIZATION [ICLR 2017]
- **Deep k-Means: Re-Training and Parameter Sharing with Harder Cluster Assignments for Compressing Deep Convolutions [ICML 2018]**
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
- <span style="color:red">Song Han, Jeff Pool, John Tran, William J. Dally:
  [Learning both Weights and Connections for Efficient Neural Networks.][ref6][NIPS 2015]</span>  
    * Alexnet prune 89% parameters，VGG prune 92.5% parameters
- Wei Wen, Chunpeng Wu, Yandan Wang, Yiran Chen, Hai Li: [Learning Structured Sparsity in Deep Neural Networks.][ref8] [NIPS 2016]  
    * Real CPU/GPU accerate with Group Sparsity/
- <span style="color:red">Yiwen Guo, Anbang Yao, Yurong Chen:
  [Dynamic Network Surgery for Efficient DNNs.][ref9] [NIPS 2016]
  </span>
    * AlexNet 17.7x compression rate
    * less training epochs
- Variational Dropout Sparsifies Deep Neural Networks [ICML 2017]
- <span style="color:red"> Pavlo Molchanov, Stephen Tyree, Tero Karras, Timo Aila, Jan Kautz
  NVIDIA:[PRUNING CONVOLUTIONAL NEURAL NETWORKS FOR RESOURCE EFFICIENT INFERENCE][ref16][ICLR 2017]</span>
- <span style="color:red">Jian-Hao Luo, Jianxin Wu, Weiyao Lin:
  [ThiNet: A Filter Level Pruning Method for Deep Neural Network Compression.][ref7][ICCV 2017]</span>  
  - layer by layer prune and fine-tune.
- **Learning Efficient Convolutional Networks through Network Slimming** [ICCV 2017]
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
- NetAdapt: Platform-Aware Neural Network Adaptation for Mobile Applications [ECCV 2018]
- **AMC: AutoML for Model Compression and Acceleration on Mobile Devices** [ECCV 2018]
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
- **ECC: Platform-Independent Energy-Constrained Deep Neural Network Compression via a Bilinear Regression Model** [CVPR 2019]
- Towards Optimal Structured CNN Pruning via Generative Adversarial Learning [CVPR 2019]
- Cascaded Projection: End-to-End Network Compression and Acceleration [CVPR 2019]
- LeGR: Filter Pruning via Learned Global Ranking [arXiv 2019] [(code)][code1]

[ref6]: https://arxiv.org/abs/1506.02626
[ref7]: https://arxiv.org/abs/1707.06342
[ref8]:http://papers.nips.cc/paper/6504-learning-structured-sparsity-in-deep-neural-networks
[ref9]:http://papers.nips.cc/paper/6165-dynamic-network-surgery-for-efficient-dnns
[ref16]: https://arxiv.org/abs/1611.06440v2



# Matrix Decomposition

- <span style="color:red"> Emily L. Denton, Wojciech Zaremba, Joan Bruna, Yann LeCun, Rob Fergus:
  [Exploiting Linear Structure Within Convolutional Networks for Efficient Evaluation.][ref10][NIPS 2014]</span>    
    * Alexnet 2.5x CPU times
- <span style="color:red"> Vadim Lebedev, Yaroslav Ganin, Maksim Rakhuba, Ivan V. Oseledets, Victor S. Lempitsky:
  [Speeding-up Convolutional Neural Networks Using Fine-tuned CP-Decomposition.][ref11][ICLR 2015] </span>  
    * CP-decom decompose a layer to four light-head layers.
- <span style="color:red">Cheng Tai, Tong Xiao, Yi Zhang, Xiaogang Wang, Weinan E:[CONVOLUTIONAL NEURAL NETWORKS WITH LOW- RANK REGULARIZATION][ref20][ICLR 2016] </span>
    * Decompose a 2d-conv to two 1d-convs
- Accelerating Very Deep Convolutional Networks for Classification and Detection [TPAMI 2016]
- **Tensor-Train Recurrent Neural Networks for Video Classification** [ICML 2017]
- Domain-adaptive deep network compression [ICCV 2017]
- Coordinating Filters for Faster Deep Neural Networks [ICCV 2017]
- <span style="color:red">Jose M. Alvarez,Mathieu Salzmann:[Compression-aware Training of Deep Networks </span> [NIPS 2017]
    * Decompose a 2d-conv to two 1d-convs, add activaiton between two layers
- Learning Compact Recurrent Neural Networks with Block-Term Tensor Decomposition [CVPR 2018]
- Wide Compression: Tensor Ring Nets [CVPR 2018]
- Extreme Network Compression via Filter Group Approximation [ECCV 2018]
- Trained Rank Pruning for Efficient Deep Neural Networks [arXiv 2018]
- Compressing Recurrent Neural Networks with Tensor Ring for Action Recognition [AAAI 2019]
- T-Net: Parametrizing Fully Convolutional Nets with a Single High-Order Tensor [CVPR 2019]


[ref10]: https://arxiv.org/abs/1404.0736
[ref11]: https://arxiv.org/abs/1412.6553
[ref20]: https://arxiv.org/abs/1511.06067
[ref25]: http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6288897



# Knowledge Distill

- **Distilling the Knowledge in a Neural Network** [2014]
- FITNETS: HINTS FOR THIN DEEP NETS [ICLR 2015]
- PAYING MORE ATTENTION TO ATTENTION:IMPROVING THE PERFORMANCE OF CONVOLUTIONAL
  NEURAL NETWORKS VIA ATTENTION TRANSFER [ICLR 2017]
- Mimicking Very Efficient Network for Object Detection [CVPR 2017]
- A Gift from Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning [CVPR 2017]
- Deep Mutual Learning [CVPR 2018]
- Data Distillation: Towards Omni-Supervised Learning [CVPR 2018]
- Quantization Mimic: Towards Very Tiny CNNfor Object Detection [ECCV 2018]
- Self-supervised Knowledge Distillation Using Singular Value Decomposition [ECCV 2018]
- KDGAN: Knowledge Distillation with Generative Adversarial Networks [NIPS 2018]
- Knowledge Distillation by On-the-Fly Native Ensemble [NIPS 2018]
- Paraphrasing Complex Network: Network Compression via Factor Transfer [NIPS 2018]
- Knowledge Transfer via Distillation of Activation Boundaries Formed by Hidden Neurons [AAAI 2019]
- Relational Knowledge Distillation [CVPR 2019]
- Snapshot Distillation: Teacher-Student Optimization in One Generation [CVPR 2019]

# Compact Model

## Efficient CNN

- SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size [arXiv 2016]
- Xception: Deep Learning with Depthwise Separable Convolutions [CVPR 2017]
- MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications [arXiv 2017]
  * depth-wise conv & point-wise conv
- ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices [CVPR 2018]
  * group-wise conv & channel shuffle
- **Shift: A Zero FLOP, Zero Parameter Alternative to Spatial Convolutions** [CVPR 2018]
- MobileNetV2: Inverted Residuals and Linear Bottlenecks [CVPR 2018]
- **ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design** [ECCV 2018]
- Sparsely Aggregated Convolutional Networks [ECCV 2018]
- Real-Time MDNet [ECCV 2018]
- ICNet for Real-Time Semantic Segmentation on High-Resolution Images [ECCV 2018]
- BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation [ECCV 2018]
- Constructing Fast Network through Deconstruction of Convolution [NIPS 2018]
- ChannelNets: Compact and Efficient Convolutional Neural Networks via Channel-Wise Convolutions [NIPS 2018]
- HetConv: Heterogeneous Kernel-Based Convolutions for Deep CNNs [CVPR 2019]
- Adaptively Connected Neural Networks [CVPR 2019]
- DFANet: Deep Feature Aggregation for Real-Time Semantic Segmentation [CVPR 2019]
- All You Need is a Few Shifts: Designing Efficient Convolutional Neural Networks for Image Classification [CVPR 2019]

## Efficient RNN variants

- QUASI-RECURRENT NEURAL NETWORKS [ICLR 2017]
- Simple Recurrent Units for Highly Parallelizable Recurrence [EMNLP 2018]
- Fully Neural Network Based Speech Recognition on Mobile and Embedded Devices [NIPS 2018]

## NAS

- DARTS: DIFFERENTIABLE ARCHITECTURE SEARCH [ICLR 2019]

[code1]: https://github.com/cmu-enyac/LeGR
