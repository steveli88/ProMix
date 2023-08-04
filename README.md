# ProMix: Combating Label Noise via Maximizing Clean Sample Utility

This is the [PyTorch](http://pytorch.org/) implementation of our IJCAI 2023 paper [ProMix](https://arxiv.org/abs/2207.10276). A previous version at the `LMNL_challenge` branch won the [1st Learning and Mining with Noisy Labels Challenge](http://competition.noisylabels.com/) in IJCAI-ECAI 2022.

**Title:** ProMix: Combating Label Noise via Maximizing Clean Sample Utility

**Authors:** Ruixuan Xiao, Dong Yiwen, Haobo Wang, Lei Feng, Runze Wu, Gang Chen, Junbo Zhao

**Affliations:** Zhejiang University, Nanyang Technological University, NetEase Fuxi AI Lab



### Framework
![Framework](./resources/framework.png)



### Main Results on CIFAR-10/100

![result_cf](./resources/result_cf.png)



### Main Results on CIFAR-N

![result_cfn](./resources/result_cfn.png)



### Usage

After creating a virtual environment, run

```
pip install -r requirements.txt
```

We provide the shell codes for model training in the `run.sh` file. Please download the source data of CIFAR-10/100 and the noise file of CIFAR-N following [Learning with Noisy Labels Revisited: A Study Using Real-World Human Annotations](https://github.com/UCSC-REAL/cifar-10-100n) and put them under the `data` folder.





###  Acknowledgement
This paper is supported by [Netease Youling Crowdsourcing Platform](https://fuxi.163.com). As the importance of data continues rising, Netease Youling Crowdsourcing Platform is dedicated to utilizing various advanced algorithms to provide high-quality, low-noise labeled samples. Feel free to contact us for more information.

