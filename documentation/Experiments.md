## Introduction

+ You may find this document helpful if you:

  + want to evaluate Heimdall and reproduce the evaluation results shown in Heimdall's paper with least burden.
  + want to know in-depth how Heimdall works and study the source code.
  + ...

  Then, please go through the following documents now!

+ Briefly speaking, Heimdall conducts comprehensive experiments including: 

  1. [Heimdall Data Science Pipeline](./1_heimdall_pipeline.md): Guideline to run Heimdall's data science pipeline of replayed traces analysis, correct labeling, feature extraction and selection, model training. In the end, you will evaluate the Heimdall model's performance on various metrics.

  2. [Joint Inference](./2_joint_inference.md): Guideline of how to run source code of Heimdall's implementation of joint-inference, and step-by-step commands to produce our evaluation results related to joint-inference in Paper Section 8.7. 

  5. [Client Level Integration](./3_client_level_integration.md): Heimdall is deployed in Client Level, and this doc shows how to replayed IO workload with the help of Heimdall in client side. Following this document will quickly produce the results of large-scale experiment in Paper Section 8.2. 

  6. [Kernel Level Integration](./4_kernel_level_integration.md): Besides to be deployed in Client and Application, we also deploy Heimdall into Linux-6.0.0. Please follow this doc to learn how to run Heimdall in kernel. This experiment corresponds to Paper Section 8.4.

+ Having a quick glance of above experiments, you may access the specific documents for each experiment by clicking the above links. Hope you have a smooth and fruitful experience with your experiments. :) 