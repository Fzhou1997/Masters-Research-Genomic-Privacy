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

The code is written in Python and utilizes the PyTorch deep learning framework.
The data is derived from the 2016 iDash Privacy and Security Workshop challenge,
which focused on the protection of genomic data sharing through Beacon services.

The repository is organized as follows:

- `data/`: contains the preprocessed genomic data used for training, evaluation, and testing of the models
- `models/`: contains the trained LSTM models state dictionaries for membership inference attacks
- `notebooks/`: contains driver Jupyter notebooks for model creation and performance landscape analysis
- `plots/`: contains visualizations of the model metrics and performance landscapes
- `src/`: contains the Python source code for the statistical and LSTM models and various utility functions

This project was supervised by Dr. Rajagopal Venkatesaramani at Northeastern University.
This is an ongoing project, and the repository will be reorganized and updated with additional code, data, and models as
the research progresses. Additional information about the project, including the methodology, results, and discussions,
can be found in the [dissertation](https://drive.google.com/file/d/1wBkfEVtdx_zHf4yVOJzfIOf3wa863qTA/view?usp=sharing).

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

<H2>Key Features</H2>

- Implements a novel deep learning approach to membership inference attacks on genomic data, addressing the limitations
  of traditional statistical methods.
- Provides a flexible and configurable CNN-LSTM-based hybrid architecture for capturing linkage disequilibrium
  dependencies in genomic sequences.
- Utilizes a real-world genomic dataset from the iDash Privacy and Security Workshop challenge.
- Includes baseline implementations of likelihood ratio test-based statistical methods for comparison.
- Offers insights into the privacy risks associated with federated genomic data analysis and the potential of deep
  learning for enhancing privacy-preserving techniques.

<H2>Installation</H2>

To run the code in this repository, you will need to have Python 3.10 or 3.11 installed on your system.
You will also need to install the required Python packages listed in the `requirements.txt` file.
You can install these packages using the following command:

```bash
pip install -r requirements.txt
```

<H2>Usage</H2>

All driver code for training, evaluating, and testing the LSTM models is provided in the Jupyter notebooks located in
the `notebooks/` directory. You can run these notebooks using JupyterLab or Jupyter Notebook.

