# DGraphormer-SleepNet

Advancing Sleep Stages Classification through a Dual-Graphormer Approach, which is improved from StAGN(https://github.com/Chen-Junyang-cn/StAGN/tree/main).

![model_structure](./model_structure.jpg)

# Environment
<pre>
python 3.9
tensorflow 2.11
cuda 11.1
</pre>

# Dataset
The ISRUC dataset can be downloaded from website: https://sleeptight.isr.uc.pt

# Preprocess
Run <code>preprocess.py</code> to pre-process the rawdata.
<p><code>python preprocess.py</code></p>

# Train model
You can change the input data path and run. Note that the output from MSFE is the DGraphormer-SleepNet's input.
<p><code>python MSFE.py</code></p>
<p><code>python DGraphormer-SleepNet.py</code></p>

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
