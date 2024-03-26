# Classification of Highly Interacting Regions of the Genome with Explainable AI

## Introduction

The human body consists of 37.2 trillion cells. All cells in our bodies have the same genomic content, but they have various capabilities and phenotypes due to the diverse proteins synthesized by the cell, accomplished through the regulation of the expression of different genes. Proteins generated from distinct sets of active genes in cells determine these features. Gene expression refers to the process of promoting or suppressing the expression of specific genes. Certain sections of the genome are extremely interacting and are generally in charge of a collection of genes, termed Topologically Associating Domains (TADs). Highly Interacting Regions (HIRs) are self-interacting domains found in the 3D genomic organization. Disruption of HIRs can result in altered gene expression, linked to genetic diseases and cancer. In a diseased cell, these regions may shift, resulting in the activation of different genes and the synthesis of incorrect proteins.

For a long time, chromosomal conformation capture-based techniques have been practiced to capture the 3D organization configuration of the genome. With advancements in sequencing, 3C-based experiments coupled with high-throughput sequencing resulted in Hi-C experimental procedures to identify genome-wide chromatin interactions. Hi-C is a technique used to detect gene proximity and chromosomal rearrangements. HIRs were discovered for the first time in 2D chromatin interaction maps using Hi-C and 5C data from populated cells as interacting squares along the diagonal, representing local contacts. They are crucial in limiting promoter–enhancer interactions. Their boundaries are preferentially stable across cell types, with only a proportion displaying cell-type specificity. HIRs record the local connections between different genomic profiles. They assist us in determining which genome regions are nearby, which helps to infer the 3D organization of the genome.

To discover what types of patterns or mutations result in a diseased condition, as compared to normal, we will use DeepSHAP, which provides information on precise patterns of the DNA structure. We will also make use of deep learning to distinguish HIRs and their boundaries and determine which traits in these regions make them more interactive, allowing us to obtain a deeper knowledge of how these specific regions influence gene regulation.

## Motivation

Our goal is to identify sequence signatures, classify them into HIRs or non-HIRs. We would also investigate these sequences present in the boundaries of the HIR regions, which would allow us to classify genomic sequences as potentially interacting regions or non-interacting regions and determine which genomic sequence properties are influential in the transformation of a cell into a diseased cell.

## Objectives

1. Learn how to model biological sequences and DNA data, and find patterns in them.
2. Efficiently use deep learning algorithms to distinguish HIRs and their boundaries.
3. Classify genomic sequences as potential HIRs or non-HIRs.
4. Identify the genomic sequence properties that influence whether a region is a HIR or not.
5. Understand how HIRs influence gene regulation and contribute to disease development.
6. Use explainable AI methods to interpret the important sequence patterns for HIR classification.

## Methodology

### Data Preparation

We referred the paper - "Sub-kb Hi-C in D. melanogaster reveals conserved characteristics of TADs between insect and mammalian cells." We downloaded the data files with domain coordinates from the National Center for Biotechnology Information (NCBI). To get the coordinates of the left and right boundaries, we took shifts of 200 and 500 base pairs and generated BED files accordingly. We then used the BEDTools utility `2BitToFa` to generate FASTA files for the sequences, which were then converted into a suitable format for processing.

### Statistical Modeling: Markov Models

We implemented Markov models to classify sequences as HIRs or non-HIRs based on their probability of belonging to each class. We trained separate Markov models on the HIR and non-HIR sequences, and classified test sequences by comparing their log-odds scores from the two models.

We explored three scenarios:

1. Markov models on dummy data with embedded HIRs.
2. Markov models on dummy data with embedded HIRs generated from JASPAR probability weight matrices.
3. Markov models with cross-validation on real data from Drosophila.

We also incorporated DNA complements in the training data to improve the classification performance.

### Deep Learning

We trained deep learning models, specifically Convolutional Neural Networks (CNNs), to classify HIRs and non-HIRs using the sequence data. The CNN architecture consisted of multiple convolutional, batch normalization, and max-pooling layers, followed by dense layers for classification.

We performed cross-validation on both simulated and real data from Drosophila to evaluate the model's performance. We also conducted Monte Carlo simulations to assess the model's robustness.

### Explainable AI: DeepSHAP

To interpret the important sequence patterns that influenced the deep learning model's predictions, we applied the DeepSHAP technique. DeepSHAP is based on Shapley Additive Explanations (SHAP), a game-theoretic approach to explaining machine learning models.

We generated various visualizations, such as summary plots, decision plots, individual force plots, and collective force plots, to understand the effects of different nucleotides (A, C, G, T) on the model's predictions.

### Evaluation

We used several metrics and visualizations to evaluate the performance of the classification models, including:

- Accuracy
- Area Under the Receiver Operating Characteristic (ROC) Curve
- Precision-Recall Curves
- Confusion Matrices

## Results

The deep learning models outperformed traditional statistical methods (Markov models) in classifying HIRs and non-HIRs based on the DNA sequence data. The CNN models achieved high accuracy and area under the ROC curve (AUC) on both simulated and real data.

The explainable AI methods like DeepSHAP provided insights into the sequence patterns that were important for the classification, potentially revealing the genomic determinants of HIRs. The visualizations generated by DeepSHAP helped identify the effects of different nucleotides on the model's predictions, shedding light on the sequence properties that influence whether a region is a HIR or not.

## Future Work

1. Investigate the biological mechanisms by which the identified sequence patterns influence HIR formation and gene regulation. Collaborate with domain experts to validate and interpret the findings.

2. Develop generative models, such as Generative Adversarial Networks (GANs) or Variational Autoencoders (VAEs), to simulate HIR sequences and study their effects on gene expression.

3. Extend the approach to study HIRs in different cell types and diseases, such as cancer. Explore how HIR disruptions contribute to disease development and identify potential therapeutic targets.

4. Incorporate additional genomic and epigenomic data, such as chromatin accessibility, histone modifications, and transcription factor binding sites, to improve the classification and interpretability of HIRs.

5. Explore other explainable AI techniques, such as LIME (Local Interpretable Model-Agnostic Explanations) or Integrated Gradients, to gain further insights into the model's decision-making process.

## Contributors

- Shreya Pawaskar
- Radhika Sethi
- Aanchal Tulsiani

This project was carried out under the guidance of Dr. Leelavati Narlikar at IISER Pune as the final year capstone project for the students' Bachelor's degree in Computer Engineering from MKSSS's Cummins College of Engineering for Women, Pune.
