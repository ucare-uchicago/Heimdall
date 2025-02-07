## Experiment Heimdall-Pipeline

---

### Experiment Description

+ **Introduction:** Heimdall explored extensive data science methodologies and ultimately delivered an efficient DS solution for IO latency reduction. This experiment presents our final DS solution and provides the procedures through which Heimdall analyzes the replayed traces, conducts correct labeling, extracts & selects the feature, and trains the model. This experiment follows heimdall's training pipeline, utilizing three provided traces to present the evaluation results.  These evaluation results include comprehensive metrics on testing set (acc, precision, f1-score, ROC-AUC, etc.) and an estimation of IO latency reduction achieved using Heimdall model.
+ **Experiment Goal:**
  1. Demonstrate Heimdall's pipeline of analyzing the replayed traces, conducting correct labeling, extracting & selecting the feature, and model training.
  2. Evaluate Heimdall's trained model on comprehensive metrics (acc, precision, f1-score, ROC-AUC, etc.).
+ **Code:** Code related to this experiment are all included in directory: `Heimdall/ds_pipeline`.
+ **Experiment Duration:** 15 mins. 
+ **Recommended Testbed:** [Chameleon](https://www.chameleoncloud.org/) `storage_hierarchy` node under [CHI@TACC](https://chi.tacc.chameleoncloud.org/project/leases/). 

The experiment will begin with `Testbed Reservation`.

---

### Testbed Reservation

+ We conduct this experiment on [Chameleon](https://www.chameleoncloud.org/) node and recommend you also conduct this experiment on Chameleon or similar testbeds for consistency and comparability. 

+ Please follow [testbed reservation guideline](./testbed_reservation.md) to reserve a `storage_hierarchy` node at CHI@TACC site.

---

### Environment Setup

+ Next, we set up the environment on testbed:

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

### Replay Traces

+ Our first step is to replay traces without the help of Heimdall and analyze the replayed results. (The replayed results will be utilized for Heimdall model training in later steps)

+ Compile the trace replayer first:

  ```bash
  cd $HEIMDALL/ds_pipeline/script/trace_replayer
  gcc io_replayer.c -o io_replayer -lpthread
  ```

+ Then, leverage the trace replay to replay on 3 sample traces we provided (suppose using `/dev/nvme1n1`):

  + You can replay all those traces by one command:

    ```bash
    sudo ./replay.sh -user $USER -device /dev/nvme1n1 -dir $HEIMDALL/ds_pipeline/data/raw_data/ -pattern "*trace" -output_dir $HEIMDALL/ds_pipeline/data/profile_data/
    ```

  + Or, replay a specific trace by:

    ```bash
    sudo ./replay.sh -user $USER -device /dev/nvme1n1 -file $HEIMDALL/ds_pipeline/data/raw_data/msr.cut.per_10k.rw_20_80.105.trace -output_dir $HEIMDALL/ds_pipeline/data/profile_data/
    ```

+ The replayed results including raw results and aggregated statistics will be store as `$HEIMDALL/ds_pipeline/data/profile_data/nvme1n1/sample.trace` and `$HEIMDALL/ds_pipeline/data/profile_data/nvme1n1/sample.trace.stats`.

---

### Replayed Traces Analysis

+ Next, we investigate deeper into the traces we replayed by:

  ```bash
  cd $HEIMDALL/ds_pipeline/script/trace_analyzer/
  
  # Alternative 1: [Run on specific trace]
  ./analyze_trace_profile.py -file $HEIMDALL/ds_pipeline/data/profile_data/nvme1n1/msr.cut.per_10k.rw_20_80.105.trace
  
  # Alternative 2: [Run on all traces in directory]
  ./analyze_trace_profile.py -files $HEIMDALL/ds_pipeline/data/profile_data/nvme1n1/*cut*trace
  ```

+ It will produce a profound characteristics profiling of replayed traces as part of our analysis. The result is stored as `/mnt/extra/Heimdall/ds_pipeline/data/profile_data/nvme1n1/sample_trace.png`. 

---

### Correct Data Labeling

+ Now, we move forward to building a high-quality dataset for heimdall's model training. The first step is to leverage our tail algorithm to correctly label the dataset. You can run it by:

  ```bash
  cd $HEIMDALL/ds_pipeline/script/tail_analyzer/
  
  # Alternative 1: [Run on specific trace]
  ./tail_analyzer.py -file $HEIMDALL/ds_pipeline/data/profile_data/nvme1n1/msr.cut.per_10k.rw_20_80.105.trace
  
  # Alternative 2: [Run on all traces in directory]
  ./tail_analyzer.py -files $HEIMDALL/ds_pipeline/data/profile_data/nvme1n1/*cut*trace
  ```

---

### Feature Extraction and Selection

+ After correct labeling, feature engineering is conducted by:

  ```bash
  cd $HEIMDALL/ds_pipeline/experiment/per_io_inference/feature_extractor/
  
  # Alternative 1: [Run on specific trace]
  ./feat_v6.py -file $HEIMDALL/ds_pipeline/data/dataset/nvme1n1/msr.cut.per_10k.rw_20_80.105/profile_v1.labeled
  
  # Alternative 2: [Run on all traces in directory]
  ./feat_v6.py -files $HEIMDALL/ds_pipeline/data/dataset/nvme1n1/*cut*/*v1*labeled
  ```

---

### Model Training

+ The above steps analyze the raw data in a deep manner and intend to construct a dataset with high quality. Finally, we train Heimdall model using this resulted dataset. The training commands are:

  ```bash
  cd $HEIMDALL/ds_pipeline/experiment/per_io_inference/model/
  
  # Alternative 1: [Train and evaluate on specific dataset]
  ./train_and_eval.py -model flashnet_binary_nn -dataset $HEIMDALL/ds_pipeline/data/dataset/nvme1n1/msr.cut.per_10k.rw_20_80.105/profile_v1.feat_v6.readonly.dataset -train_eval_split 50_50
  
  # Alternative 2: [Train and evaluate on all datasets in directory]
  ./train_and_eval.py -model flashnet_binary_nn -datasets $HEIMDALL/ds_pipeline/data/dataset/nvme1n1/*/profile_v1.feat_v6.readonly.dataset -train_eval_split 50_50

+ Exception for training, this step will also conduct testing, which evaluates our model on various angles below:

  1. **Evaluation metrics on testing set**. Includes accuracy, precision, recall, f1-score, ROC-AUC and so on. 
     + Store in stats files like: `/mnt/heimdall-exp/Heimdall/ds_pipeline/data/dataset/nvme1n1/msr.cut.per_10k.rw_20_80.105/profile_v1.feat_v6.readonly/flashnet_binary_nn/split_50_50/128_16/eval.stats`

  2. **A Confusion Metrics & Estimated IO Latency CDF with the help of Heimdall**
     + Store in figures like: `/mnt/heimdall-exp/Heimdall/ds_pipeline/data/dataset/nvme1n1/msr.cut.per_10k.rw_20_80.105/profile_v1.feat_v6.readonly/flashnet_binary_nn/split_50_50/128_16/eval.png`
