<H1 align="middle"> Deep Learning LSTM for Genomic Privacy </H1>

<p align="middle">
    <strong>
        Exploring the application deep learning LSTM networks for re-identification of federated genomic data
    </strong>
</p>

<br>
<H2>Description</H2>

This repository contains the code, data, and models for my Master's Thesis project,
which explores the application of Long Short-Term Memory (LSTM) networks for to membership inference attacks on
federated genomic data. The project aims to address the limitations of previous statistical methods, such as the
likelihood ratio test, by leveraging deep learning to capture linkage disequilibrium (LD) dependencies in genomic data.

This project was supervised by Dr. Rajagopal Venkatesaramani at Northeastern University.
This is an ongoing project, and the repository will be reorganized and updated with additional code, data, and models as
the research progresses. Additional information about the project, including the methodology, results, and discussions,
can be found in the [dissertation](https://drive.google.com/file/d/1wBkfEVtdx_zHf4yVOJzfIOf3wa863qTA/view?usp=sharing).

<br>
<H2>Motivation</H2>

Federated genomic data analysis has emerged as a critical tool for enabling collaborative research while addressing
privacy concerns. However, traditional methods for genomic data anonymization, such as allele frequency
summary statistics and the Beacon Protocol, have been shown to be vulnerable to membership inference attacks (MIAs).  
These attacks exploit subtle patterns and correlations within the data to identify whether a specific individual's data
is included in a dataset.

Deep learning models, particularly Long Short-Term Memory (LSTM) networks, have demonstrated their ability to capture
complex dependencies in genomic sequences. This project aims to leverage the power of LSTM networks to develop more
accurate MIAs, thereby enhancing our understanding of the privacy risks associated with federated genomic data analysis.

By developing more effective MIAs, we can gain deeper insights into the vulnerabilities of existing privacy-preserving
techniques and inform the design of more robust mechanisms for protecting genomic data. This is crucial for fostering
responsible data sharing and ensuring the continued advancement of genomic research while upholding the highest
standards of data protection.

<br>
<H2>Key Features</H2>

- Implements a novel deep learning approach to membership inference attacks on genomic data, addressing the limitations
  of traditional statistical methods.
- Provides a flexible and configurable CNN-LSTM-based hybrid architecture for capturing linkage disequilibrium
  dependencies in genomic sequences.
- Utilizes a real-world genomic dataset from the iDash Privacy and Security Workshop challenge.
- Includes baseline implementations of likelihood ratio test-based statistical methods for comparison.
- Offers insights into the privacy risks associated with federated genomic data analysis and the potential of deep
  learning for enhancing privacy-preserving techniques.

<br>
<H2>Implementation</H2>

The project is implemented in Python using the PyTorch deep learning library.

<H3>Dataset</H3>

The dataset used in this project is derived from the 2016 iDash Privacy and Security Workshop challenge, which focused
on the protection of genomic data sharing through Beacon services. The dataset consists of over 1.3 million single
nucleotide variants (SNVs) from chromosome 10 for 800 individuals. The dataset is divided into two subsets: 400
individuals whose data are included in the federated set and 400 whose data are excluded. The genomic sequencing error
rate is δ=10^−6.

<H3>Model Architecture</H3>

The core model architecture features a hybrid CNN-LSTM design, including three Conv1D layers for motif detection and
four LSTM layers for capturing sequential dependencies like linkage disequilibrium. Regularization techniques such as
dropout and batch normalization were employed to mitigate overfitting. The model architecture is flexible and can be
configured to use different numbers of layers and hidden units. These hyperparameters are tuned using a random search
strategy.

<H3>Results Summary</H3>

The following table summarizes the test-set metrics of the best performing LSTM-based attacker models on both the Beacon
and AAF datasets:

| Dataset | Accuracy | AUROC  | F1-Score |
|---------|----------|--------|----------|
| Beacon  | 0.5917   | 0.5916 | 0.5664   |
| AAF     | 0.6000   | 0.6000 | 0.6667   |

The following table summarizes the metrics of the likelihood ratio test-based attacker models on both the Beacon and AAF
datasets:

| Dataset | Accuracy | AUROC | F1-Score |
|---------|----------|-------|----------|
| Beacon  | 0.97     | 1.00  | 0.98     |
| AAF     | 0.89     | 0.96  | 0.89     |

While the lstm-based models did not surpass the likelihood ratio test (LRT) baseline, it provides a foundation for
future exploration of deep learning in genomic privacy tasks.

The following table summarizes some hyperparameters of the best performing LSTM-based attacker on the beacon dataset:

| Dataset | Model ID | Conv1D Stack Depth | LSTM Stack Depth | Linear Stack Depth |
|---------|----------|--------------------|------------------|--------------------|
| Beacon  | 11062353 | 3                  | 4                | 1                  |
| AAF     | 12052257 | 3                  | 1                | 1                  |

Key visualizations, including ROC curves and hyperparameter performance landscapes, are available in the `plots/`
directory.

<H3>Future Directions</H3>

The current LSTM architecture has several limitations. First, the model is limited by the amount of data that is
available for training. Second, the model is not able to fully capture the complexity of linkage disequilibrium
patterns. Finally, the model is itself susceptible to membership inference attacks.

Future research should focus on developing more specialized LSTM architectures for genomic privacy tasks. Additionally,
efforts should be made to collect and curate larger and more representative genomic datasets. Finally, research should
explore the use of adversarial training to develop more robust privacy-preserving mechanisms.

<br>
<H2>Repository Structure</H2>

```
.
├── data/idash                  # iDash Privacy and Security Workshop genomic dataset
├── models                      # Trained models and model hyperparameters and metrics table
├── notebooks                   # Jupyter notebooks for training, evaluation, testing, and hyperparameter optimization
├── plots                       # Output plots and visualizations
├── src                         # Source code for data preprocessing, model training, and evaluation
│   ├── utils_attacker_lrt      # Likelihood ratio test-based attacker implementations
│   ├── utils_attacker_lstm     # LSTM-based attacker implementations
│   ├── utils_io                # Input/output utilities for loading data
│   ├── utils_plot              # Plotting utilities for visualizing results
│   ├── utils_random            # Random number generation utilities
│   └── utils_torch             # PyTorch data and module utilities
├── requirements.txt            # Python package requirements
└── README.md                   # Project overview and repository structure
```

<br>
<H2>Installation</H2>

To run the code in this repository, you will need to have Python 3.10 or 3.11 installed on your system.
You will also need to install the required Python packages listed in the `requirements.txt` file.
You can install these packages using the following command:

```bash
pip install -r requirements.txt
```

<br>
<H2>Usage</H2>

All driver code for training, evaluating, and testing the LSTM models is provided in the Jupyter notebooks located in
the `notebooks/` directory. You can run these notebooks using JupyterLab or Jupyter Notebook.

