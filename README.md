# Structural Optimization of Factor Graphs for Symbol Detection via Continuous Clustering and Machine Learning

This repo provides a demo for [1], which demonstrates the training and evaluation of the structural optimization of factor graphs.

[1] L. Rapp, L. Schmid, A. Rode, and L. Schmalen, “Structural Optimization of Factor Graphs for Symbol Detection via Continuous Clustering and Machine Learning,” 2023, arXiv:2211.11406. [Online]. Available: https://arxiv.org/abs/2211.11406.

**Abstract:**
We propose a novel method to optimize the structure of factor graphs for graph-based inference. As an example inference task, we consider symbol detection on linear inter-symbol interference channels.  The factor graph framework has the potential to yield low-complexity symbol detectors. However, the sum-product algorithm on cyclic factor graphs is suboptimal and its performance is highly sensitive to the underlying graph. Therefore, we optimize the structure of the underlying factor graphs in an end-to-end manner using machine learning. For that purpose, we transform the structural optimization into a clustering problem of low-degree factor nodes that incorporates the known channel model into the optimization. Furthermore, we study the combination of this approach with neural belief propagation, yielding near-maximum a posteriori symbol detection performance for specific channels.

---

This work has received funding in part from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (grant agreement No. 101001899) and in part from the German Federal Ministry of Education and Research (BMBF) within the project Open6GHub (grant agreement 16KISK010).