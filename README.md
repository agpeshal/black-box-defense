# Adversarial Defense



Experiments on CIFAR10 trained on ResNet18 to defend the network against state-of-the-art HopSkipJumpAttack (Chen *et al*).

### File Structure

```bash
├── deepfool.py		# DeepFool attack
├── helpers.py		# helper functions (primarliy for debugging)
├── hsja.py		# HSJA attack
├── models
│   ├── base
│   │   └── ckpt.pth	# pretrained standard model
│   └── curvature
│       ├── CURE.py	# main defense implementation
│       └── robust.pth	# defended model
├── resnet.py		# Resnet architecture definitions
├── train.py		# train to defend against attacks
└── utils
    └── utils.py	# utility functions
```



### Attack

Decision based Hop Skip Jump Attack (Chen *et al*)

![](images/hsja.png)

### Defense

The idea is to analyze HSJA assumptions and increase the curvature of the boundary inspired by Curvature regularization (Moosavi *et al*) 

```bash
python train.py --epochs 40 --batch_size 128 --lr 0.0001
```

The above command would re-train the model with the aim to increase boundary curvature hoping the attack to fail

![](images/cure.png)
*Image courtesy: Moosavi et al*

### References

- Chen, Jianbo, Michael I. Jordan, and Martin J. Wainwright. "[HopSkipJumpAttack: A query-efficient decision-based attack](https://arxiv.org/pdf/1904.02144.pdf)." *IEEE Symposium on Security and Privacy (SP). IEEE, 2020.*
- Moosavi-Dezfooli, Seyed-Mohsen, et al. "[Robustness via curvature regularization, and vice versa](https://openaccess.thecvf.com/content_CVPR_2019/papers/Moosavi-Dezfooli_Robustness_via_Curvature_Regularization_and_Vice_Versa_CVPR_2019_paper.pdf)" *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2019.
