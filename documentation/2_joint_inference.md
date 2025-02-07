## Experiment Joint Inference

---

### Experiment Description

+ **Introduction:** To enhance throughput under intensive IO submissions, we implement **joint-inference** within Heimdall. Joint-inference involves adapting the model to be able to take features from up to P parallel IO operations, allowing it to make one inference on behalf of all of them.

  Joint inference of merging several inferences into one will improve throughput. While the critical evaluation question is: Will this approach compromise the accuracy of our model? To address this, we include the source code for joint inference together with several sample datasets in this experiment. We aims at evaluating the accuracy of the model when deployed with joint inference.

+ **Experiment Goal:**

  1. Evaluate the model on testing set when deployed with joint inference.

+ **Code:** Code related to this experiment is included in directory: `Heimdall/ds_pipeline/experiment/joint_inference`.

+ **Experiment Duration:** 5 mins. 

+ **Recommended Testbed:** [Chameleon](https://www.chameleoncloud.org/) `storage_hierarchy` node under [CHI@TACC](https://chi.tacc.chameleoncloud.org/project/leases/). 

The experiment will begin with `Testbed Reservation`.

---

### Testbed Reservation

+ We conduct this experiment on [Chameleon](https://www.chameleoncloud.org/) node and recommend you also conduct this experiment on Chameleon or similar testbeds for consistency and comparability. 

+ Please follow [testbed reservation guideline](./testbed_reservation.md) to reserve a `storage_hierarchy` node at CHI@TACC site. (The first experiment, referred to as [heimdall_pipeline](./1_heimdall_pipeline.md), utilizes the same node as well. If you have previously reserved a node for this purpose, you may continue to use the same node and skip this part.)

---

### Environment Setup

(If you have done this for [heimdall_pipeline](./1_heimdall_pipeline.md), you may skip this part of env setup.)

+ After accessing to testbed, we set up the environment on testbed:

  ```bash
  # If you already do this in other experiments, skip this
  sudo mkdir -p /mnt/heimdall-exp    
  sudo chown $USER -R /mnt
  cd /mnt/heimdall-exp
  git clone https://github.com/ucare-uchicago/Heimdall.git
  ```

  ```bash
  cd /mnt/heimdall-exp/Heimdall
  echo 'export HEIMDALL='$(pwd) >> ~/.bashrc
  source  ~/.bashrc
  ```

+ Dependencies installation:

  ```bash
  pip3 install numpy
  pip3 install pandas
  pip3 install matplotlib
  pip3 install scikit-learn
  pip3 install statsmodels
  pip3 install tensorflow
  pip3 install auto-sklearn
  pip3 install bokeh
  pip3 install seaborn
  ```

---

### Dataset Reorganization and Labeling

+ Since joint inference adapts the model to combine multiple inferences into one, the input features of the model must be reorganized accordingly. Therefore, in this step, we amalgamate multiple samples into one sample with extended features and an aligned label by following commands.

+ Argument `-batch_size 2` means we set joint-inference to merge 2 inferences into 1:

  ```bash
  cd $HEIMDALL/ds_pipeline/experiment/joint_inference/batch_analyzer/
  
  # Joint Inference with size 2 (Merge 2 inferences into 1)
  ./joint_inference_feat_v6.py -batch_size 2 -files $HEIMDALL/ds_pipeline/data/dataset/nvme1n1/*.cut.*/profile_v1.feat_v6.readonly.dataset
  
  # Joint Inference with size 9
  ./joint_inference_feat_v6.py -batch_size 9 -files $HEIMDALL/ds_pipeline/data/dataset/nvme1n1/*.cut.*/profile_v1.feat_v6.readonly.dataset
  ```

---

### Model Training & Evaluation

+ Train the model deployed with joint-inference and evaluate the accuracy on testing set:

  ```bash
  cd $HEIMDALL/ds_pipeline/experiment/joint_inference/model/
  
  # Joint Inference with size 2 (Merge 2 inferences into 1)
  ./train_and_eval.py -model flashnet_binary_nn_joint -datasets $HEIMDALL/ds_pipeline/data/dataset/nvme1n1/*.cut.*/profile_v1.feat_v6.readonly.batch_2.dataset -train_eval_split 50_50
  
  # Joint Inference with size 9
  ./train_and_eval.py -model flashnet_binary_nn_joint -datasets $HEIMDALL/ds_pipeline/data/dataset/nvme1n1/*.cut.*/profile_v1.feat_v6.readonly.batch_9.dataset -train_eval_split 50_50
  ```

+ Output evaluation results:

  1. **Evaluation metrics on testing set**. Includes accuracy, precision, recall, f1-score, ROC-AUC and so on. 
     + Store in stats files like: `Heimdall/ds_pipeline/data/dataset/nvme1n1/msr.cut.per_10k.rw_40_60.1370/profile_v1.feat_v6.readonly.batch_9/flashnet_binary_nn_joint/split_50_50/eval.stats`

  2. **A Confusion Metrics & Estimated IO Latency CDF with the help of Heimdall (joint-inference deployed)**
     + Store in figures like: `Heimdall/ds_pipeline/data/dataset/nvme1n1/msr.cut.per_10k.most_thpt_rand_iops.1006/profile_v1.feat_v6.readonly.batch_9/flashnet_binary_nn_joint/split_50_50/eval.png`

