# RTlogD

Code for "Prediction Enhanced by Transferring Knowledge from Chromatographic Retention Time, Microscopic pKa and logP".  Please read our paper https://doi.org/10.1186/s13321-023-00754-4 for more detials.

# Abstract of article
Lipophilicity is a fundamental physical property that significantly affects various aspects of drug behavior, including solubility, permeability, metabolism, distribution, protein binding, and toxicity. Accurate prediction of lipophilicity, measured by the logD7.4 value (the distribution coefficient between n-octanol and buffer at physiological pH 7.4), is crucial for successful drug discovery and design. In this study, we present a novel in silico logD7.4 prediction model called RTlogD. Our model combined a pre-training model on a chromatographic retention time dataset with a fine-tuning model that includes multitasks of logD and logP. We also incorporated microscopic acidic pKa and basic pKa into atomic features. Our model exhibited superior performance compared to existing tools and models, such as Instant Jchem, ADMETlab2.0, PCFE, FP-ADMET and ALOGPS. We conducted case studies and analyses to validate the strategies proposed in this paper.
![image](https://github.com/myzhengSIMM/RTlogD/assets/150652802/472f25a8-4117-4325-8b5c-70c4b69b98ce)

# Installation

We recommend to use Mamba to create the environment for RTlogD.

This is not a strict requirement. You can use conda instead, but it offers a quicker way for installation. If you want to use it, you can install mamba in https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html.

```
mamba create -n RTlogD python=3.9
mamba activate RTlogD
pip install torch==1.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
mamba install dgl-cuda11.3==0.9.0 -c dglteam
mamba install dgllife=0.2.9
mamba install rdkit pandas tqdm psutil
git clone https://github.com/myzhengSIMM/RTlogD.git
```

# Using

# Test

#### **Reproducing results in the paper**

```python
python test_T_data.py 
```

Run the above script to get the results of RTlogD on T-data.

#### For predicting new molecules

1. Put the data into example.csv

2. ```
    python test.py 
   ```

3.  The predicted results will be shown in the results folder.

# Author

s20-wangyitian@simm.ac.cn
