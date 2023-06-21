# [Model] Prime

<!-- Insert the project banner here -->
<div align="center">
    <a href="https://github.com/ai4protein/Prime/"><img width="600px" height="auto" src="https://github.com/ai4protein/Prime/blob/main/band.png"></a>
</div>

<!-- Select some of the point info, feel free to delete -->
![PyTorch Version](https://img.shields.io/badge/dynamic/json?color=blue&label=pytorch&query=%24.pytorchVersion&url=https%3A%2F%2Fgist.githubusercontent.com/PaParaZz1/54c5c44eeb94734e276b2ed5770eba8d/raw/85b94a54933a9369f8843cc2cea3546152a75661/badges.json)
[![GitHub license](https://img.shields.io/github/license/ai4protein/Prime)](https://github.com/ai4protein/Prime/blob/main/LICENSE)

Updated on 2023.06.15



## Key Features

This repository provides the official implementation of Prime (Protein language model for Intelligent Masked pretraining and Environment (temperature) prediction).

Key features:
- OGT (Optimal Growth Temperature) prediction.
- Zero-shot mutant effect prediction.

## Links

- [Paper](https://arxiv.org/abs/2304.03780)
- [Code](https://github.com/ai4protein/Prime) 

## Details

### What is Prime?
Prime, a novel protein language model, has been developed for predicting the Optimal Growth Temperature (OGT) and enabling zero-shot prediction of protein thermostability and activity. This novel approach leverages temperature-guided language modeling.
<div align="center">
    <a href="https://"><img width="1000px" height="auto" src="https://github.com/ai4protein/Prime/blob/main/model.jpg"></a>
</div>

### The performance of Prime.

#### Performance of the OGT prediction task.
|                                                                |  RMSE | Pearson correlation coefficient | Spearman's rank correlation coefficient | Coefficient of determination |
|:--------------------------------------------------------------:|:-----:|:-------------------------------:|-----------------------------------------|------------------------------|
|                             Prime                              | 4.736 |              0.653              |                  0.598                  |             0.417            |
| [DeepET](https://onlinelibrary.wiley.com/doi/10.1002/pro.4480) | 5.985 |              0.417              |                  0.333                  |             0.069            |

#### Performance of the zero-shot mutant effect prediction.

##### TM datasets
|           | ESM-1v | ROSSETA | MSA-Transformer | Prime(No OGT task) | Prime  |     Tranception      | Prime (Fine-tuned) |
|:---------:|:------:|:-------:|:---------------:|:------------------:|:------:|:--------------------:|:------------------:|
| 1SHF-8.0  |  0.122 |  0.152  |      0.131      |       -0.028       | 0.131  |        0.247         |       0.453        |
| 1CSP-7.0  |  0.404 |  0.020  |      0.039      |       -0.009       | 0.515  |        0.203         |       0.475        |
| 1PWX-7.7  | -0.046 |  0.158  |      0.010      |       0.595        | -0.019 |        0.106         |       0.460        |
| 1PIN-7.0  |  0.503 |  0.003  |      0.362      |       -0.023       | 0.577  |        -0.138        |       0.665        |
| 2LZM-3.0  |  0.261 |  0.511  |      0.609      |       0.576        | 0.706  |        0.480         |       0.213        |
| 3UMR-7.4  |  0.488 |  0.247  |      0.386      |       0.537        | 0.523  |        0.484         |       0.451        |
| 1RGG-7.0  |  0.719 |  -0.066 |      0.212      |       0.327        | 0.725  |        0.562         |       -0.022       |
| 1BPI-7.0  |  0.527 |  0.260  |      0.389      |       0.517        | 0.448  |        0.238         |       0.595        |
| 2DRI-7.0  |  0.127 |  -0.602 |      -0.425     |       0.381        | 0.011  |        0.541         |       0.845        |
| 2LZM-5.42 |  0.564 |  0.697  |      0.442      |       0.236        | 0.552  |        -0.152        |       -0.107       |
| 1CF1-7.0  |  0.549 |  0.150  |      0.158      |       0.508        | 0.549  |        0.438         |       0.645        |
| 1AYF-8.5  |  0.518 |  -0.082 |      0.491      |       0.655        | 0.718  |        0.455         |       0.709        |
| 5dfr-7.8  |  0.568 |  0.457  |      0.425      |       0.507        | 0.439  |        0.346         |       0.839        |
| 2LZM-6.78 | -0.014 |  0.021  |      0.091      |       -0.154       | 0.070  |        0.133         |       0.372        |
| 2LZM-3.05 |  0.484 |  0.463  |      0.816      |       0.387        | 0.613  |        0.608         |       0.734        |
| 1g5a-7.0  |  0.237 |  0.140  |      0.184      |       -0.144       | 0.097  |        0.084         |       0.413        |
| 1BNI-6.3  |  0.689 |  0.421  |      0.900      |       0.758        | 0.795  |        0.682         |       0.442        |
| 2LZM-5.4  |  0.179 |  0.492  |      0.475      |       0.401        | 0.544  |        0.631         |       0.673        |
| 1LRP-7.0  |  0.798 |  -0.432 |      0.770      |       0.549        | 0.529  |        0.733         |       -0.181       |
| 1ROP-7.0  |  0.257 |  0.805  |      0.605      |       -0.048       | 0.359  |        0.446         |       0.478        |
| 2LZM-5.35 |  0.683 |  0.825  |      0.882      |       0.778        | 0.760  |        0.590         |       0.745        |
| 1XAS-6.0  | -0.259 |  0.329  |      0.203      |       0.028        | 0.014  |        0.420         |       0.832        |
|  Average  |  0.380 |  0.226  |      0.371      |       0.333        | 0.439  |        0.370         |       0.488        |

##### Activity datasets
|                      | ESM-1v | MSA-transformer | Prime (No OGT task) | Prime | Tranception | Prime (fine-tuned) |
|:--------------------:|:------:|:---------------:|:-------------------:|:-----:|:-----------:|:------------------:|
|       BG_STRSQ       |  0.598 |      0.598      |        0.686        | 0.647 |    0.580    |       0.743        |
|      BLAT_ECOLX      |  0.687 |      0.297      |        0.718        | 0.731 |    0.580    |       0.416        |
|       TIM_SULSO      |  0.616 |      0.613      |        0.627        | 0.663 |    0.572    |       0.646        |
|     P84126_THETH     |  0.578 |      0.036      |        0.572        | 0.583 |    0.398    |       0.652        |
|      UBC9_HUMAN      |  0.536 |      0.403      |        0.472        | 0.508 |    0.319    |       0.281        |
|      AMIE_PSEAE      |  0.532 |      0.556      |        0.383        | 0.345 |    0.463    |       0.548        |
| MTH3_HAEAESTABILIZED |  0.457 |      0.490      |        0.397        | 0.355 |    0.293    |       0.560        |
|     B3VI55_LIPST     |  0.265 |      0.303      |        0.247        | 0.264 |    0.189    |       0.292        |
|      KKA2_KLEPN      |  0.174 |      0.121      |        0.177        | 0.182 |    0.105    |       0.178        |
|      MK01_HUMAN      |  0.153 |      0.176      |        0.203        | 0.148 |    0.006    |       0.164        |
|      RASH_HUMAN      |  0.125 |      0.174      |        0.119        | 0.129 |    0.107    |       0.155        |
|        Average       |  0.429 |      0.343      |        0.418        | 0.414 |    0.328    |       0.421        |


### Tips
It is important to emphasize that Prime builds upon the [esm2-650M](https://github.com/facebookresearch/esm) framework. Specifically, we utilize the checkpoint provided by the framework to initialize the model parameters.


## Get Started

**Main Requirements**  
> biopython==1.81   
> numpy==1.24.3     
> pandas==2.0.2     
> scipy==1.10.1     
> tokenizers==0.13.3    
> torch==1.12.0     
> tqdm==4.65.0  
> transformers==4.30.2  

**Installation**
```bash
pip install -r requirements.txt
```

**Download Model**

[prime-base](https://drive.google.com/file/d/1sjl-0JNBr5EH5PXy6dbkcZaO50zYklGe/view)

[prime-fine-tuning-for-tm-datasets](https://drive.google.com/file/d/1jo3OMJSCNuB_To2gNjOSCqNVjmqo2dZI/view?usp=drive_link)


**Predicting OGT**
```bash
python predict_ogt.py --model_name prime-base \
--fasta ./datasets/OGT/ogt_small.fasta \
--output ogt_prediction.tsv
```


**Predicting Mutant Effect**

Using the prime-base model. (Recommended)
```shell
python predict_mutant.py --model_name prime-base \
--fasta ./datasets/TM/1CF1/1CF1.fasta \
--mutant ./datasets/TM/1CF1/1CF1-7.0.tsv \
--compute_spearman \
--output pred.tsv
```

Or using the model that fine-tuned on the homologous sequence of the proteins in the TM dataset.
```shell
python predict_mutant.py --model_name prime-tm-fine-tuning \
--fasta ./datasets/TM/1CF1/1CF1.fasta \
--mutant ./datasets/TM/1CF1/1CF1-7.0.tsv \
--compute_spearman \
--output pred.tsv
```


## 🙋‍♀️ Feedback and Contact

- [Send Email](mailto:ginnmelich@gmail.com)

## 🛡️ License

This project is under the MIT license. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgement

A lot of code is modified from [🤗 transformers](https://github.com/huggingface/transformers) and [esm](https://github.com/facebookresearch/esm).

## 📝 Citation

If you find this repository useful, please consider citing this paper:
```
@misc{tan2023,
      title={Engineering Enhanced Stability and Activity in Proteins through a Novel Temperature-Guided Language Modeling.}, 
      author={Pan Tan and Mingchen Li and Liang Zhang and Zhiqiang Hu and Liang Hong},
      year={2023},
      eprint={2304.03780},
      archivePrefix={arXiv},
      primaryClass={q-bio.QM}
}
```
