# PINNs-TF2.0
Implementation in TensorFlow 2.0 of different examples put together by Raissi _et al._ on their original publication about Physics Informed Neural Networks.

By designing a custom loss function for standard fully-connected deep neural networks, enforcing the known laws of physics governing the different setups, their work showed that it was possible to either solve or discover with surprisingly good accuracy Partial Differential Equations from noisy and scarce  data. The very kind that is widespread in real-life applications.

Don’t forget the `--recursive` flag while cloning, in order to fetch the experiments data put together by Raissi _et al_:
```console
git clone https://github.com/pierremtb/PINNs-TF2.0 --recursive
```

## Also available on Google Colab (installation-free, GPU-enabled cloud notebooks)
- [PINNs for 1D Burgers Equation (TF2.0).ipynb](https://colab.research.google.com/drive/1lo7Kf8zTb-DF_MjkO8Y07sYELnX3BNUR)

## Authors and citations
For more information, please refer to the following: (https://maziarraissi.github.io/PINNs/)

  - Raissi, Maziar, Paris Perdikaris, and George Em Karniadakis. "[Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations](https://arxiv.org/abs/1711.10561)." arXiv preprint arXiv:1711.10561 (2017).
  
  - Raissi, Maziar, Paris Perdikaris, and George Em Karniadakis. "[Physics Informed Deep Learning (Part II): Data-driven Discovery of Nonlinear Partial Differential Equations](https://arxiv.org/abs/1711.10566)." arXiv preprint arXiv:1711.10566 (2017).

```
@article{raissi2017physicsI,
    title={Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations},
    author={Raissi, Maziar and Perdikaris, Paris and Karniadakis, George Em},
    journal={arXiv preprint arXiv:1711.10561},
    year={2017}
}

@article{raissi2017physicsII,
    title={Physics Informed Deep Learning (Part II): Data-driven Discovery of Nonlinear Partial Differential Equations},
    author={Raissi, Maziar and Perdikaris, Paris and Karniadakis, George Em},
    journal={arXiv preprint arXiv:1711.10566},
    year={2017}
}
```

## License
MIT License

Copyright (c) 2019 Pierre Jacquier
