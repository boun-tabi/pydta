# PyDTA ([Arxiv Paper](https://arxiv.org/abs/2107.05556))

### Installation
It is recommended to install this library inside a virtual enviroment, since it requires many other libraries with specific versions. It is enough to run pip installer, it will install all dependencies.
```
pip3 install pydta
```

### Library Organization
#### 1. pydata.models
This module contains the definition of DTA models presented in our paper. The weak learners are also implemented in this module.

#### 2. pydata.data
This module implements data related functions and classes. You can also find a sample DTA dataset here.

#### 3. pydata.evalutation
This module contains the implementation of metrics used in the paper, which are rmse, mse, r2 and ci.

#### 4. pydata.utils
This module contains helper functions.

---

### Tutorial
You can check out the ipynb notebook named "pydata samples.ipynb" where you can find 4 sample usages of this library. 

