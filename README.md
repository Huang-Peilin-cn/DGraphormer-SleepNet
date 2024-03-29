# DGraphormer-SleepNet

DGraphormer-SleepNet: A Dual-graphormer-based Method for Sleep Stage Classification, which is improved from StAGN(https://github.com/Chen-Junyang-cn/StAGN/tree/main).

![model_structure](./model_structure.jpg)

# Citation
If you find this useful, please cite our work as follows:
<pre>
  @inproceedings{huang2024DGraphormer,
  title={DGraphormer-SleepNet: A Dual-graphormer-based Method for Sleep Stage Classification},
  author={Peilin Huang, Meiyu Qiu, Yi Liu, Bowen Zhang, Weidong Gao,and Xiaomao Fan},
  booktitle={},
  pages={},
  year={2024},
  organization={}
}
</pre>


# Dataset
The ISRUC dataset can be downloaded from website: https://sleeptight.isr.uc.pt

# Preprocess
Run <code>preprocess.py</code>
<p><code>python preprocess.py</code></p>

# Train model
You can change the input data path and run. Note that the output from MSFE is the DGraphormer-SleepNet's input.
