# FineD-Eval

Repository for EMNLP-2022 Paper (FineD-Eval: Fine-grained Automatic Dialogue-Level Evaluation)

### Installation

```bash
conda env create -f environment.yml
conda activate finedeval
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```
### Resources
Download data.zip, output.zip, and roberta_full_base.zip at <br />
https://www.dropbox.com/sh/8zyzxe53pt9zkwe/AACJRW54n-6v4btlRK7CtfhAa?dl=0 <br />
unzip the three zip files and put everything under the current folder

### Train
see bash files in scripts/train

### inference
see bash files in scripts/eval


### Compute Correlation
enter the output folder and execute the following example: <br />
```
python dialogue_compute.py --prefix coherence_base/dailydialog_coherence_123456/
```

### Cite the following if you find this repo useful;

```
@inproceedings{zhang-etal-2022-finedeval,
    title = "{F}ine{D}-{E}val: Fine-grained Automatic Dialogue-Level Evaluation",
    author = "Zhang, Chen  and
      D{'}Haro, Luis Fernando  and
      Zhang, Qiquan  and
      Friedrichs, Thomas  and
      Li, Haizhou",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    publisher = "Association for Computational Linguistics",
}
```

## Acknowledge

The implementation of this repository is modified from https://github.com/princeton-nlp/MADE

```
@inproceedings{friedman2021single,
   title={Single-dataset Experts for Multi-dataset QA},
   author={Friedman, Dan and Dodge, Ben and Chen, Danqi},
   booktitle={Empirical Methods in Natural Language Processing (EMNLP)},
   year={2021}
}
```
