## Experiment Client-Level Integration

---

### Experiment Description

+ **Introduction:** We have deployed Heimdall in various platforms, including at the **Client-Level**, Application-Level, and Kernel-Level. This particular experiment focuses on a **comprehensive and unbiased evaluation of Heimdall at the Client-Level**. To ensure a thorough and unbiased evaluation, we conducted a **large-scale evaluation** with over hundreds of random time windows ("traces") from various real-world, multi-day traces such as from Alibaba, MSR, and Tencent. We measure IO latencies during replaying these traces and finally evaluate latencies reduction brought by Heimdall against other baseline methods.
+ **Experiment Goal:**
  1. Evaluate the **average IO latencies reduction** of Heimdall in client level.
  2. Evaluate the **IO latencies across through percentiles** of Heimdall in client level.

+ **Baselines:** 
  1. Baseline: Submitting IO requests to primary device without any redirection to secondary device.
  2. Random: Submitting IO requests randomly to any one the devices.
  3. Hedging-P95: Hedging on percentile 95 latency.
  4. LinnOS Client: Another ML-based method to reduce IO latencies.
  5. LinnOS + Hedging: A combined method of LinnOS and Hedging.
+ **Traces Clarification:** For our evaluation in paper, we conduct large-scale experiments on hundreds of various and random real-world traces. However, those traces are both too large (around 13 TB) and too time-consuming (takes several days to run, even in parallel way) to be included in the artifact. Therefore, we randomly include 5 of them in artifact just for quick evaluation.

+ **Code:** Code related to this experiment is included in directory: `Heimdall/integration/client-level`.
+ **Experiment Duration:** 90 minutes (only replaying) + 120 minutes (training flashnet and linnos at same time)
+ **Recommended Testbed:** [Chameleon](https://www.chameleoncloud.org/) `storage_hierarchy` node under [CHI@TACC](https://chi.tacc.chameleoncloud.org/project/leases/). 

The experiment will begin with `Testbed Reservation`.

---

### Testbed Reservation

+ We conduct this experiment on [Chameleon](https://www.chameleoncloud.org/) node and recommend you also conduct this experiment on Chameleon or similar testbeds for consistency and comparability. 

+ Please follow [testbed reservation guideline](./testbed_reservation.md) to reserve a `storage_hierarchy` node at CHI@TACC site. (The first three experiments, referred to as [heimdall_pipeline](./1_heimdall_pipeline.md), [joint_inference](./2_joint_inference.md) and [auto_ml](./3_auto_ml.md), utilizes the same node as well. If you have previously reserved a node for this purpose, you may continue to use the same node and skip this part.)

---

### Environment Setup

(If you have done this for the experiments before, you may skip this part of env setup.)

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

### Run Baseline

+ First of all, we replay the provided traces using **Baseline** (submitting IO requests to primary device without any redirection to secondary device) and get the corresponding IO latencies. 

+ Compile the baseline by:

  ```bash
  cd $HEIMDALL/integration/client-level/experiment/baseline
  
  make
  ```

+ Replay traces using the baseline method (Suppose the two devices we use are `/dev/nvme0n1` and `/dev/nvme1n1`. We should keep using consistent devices in following replaying for comparability):

  ```bash
  cd $HEIMDALL/integration/client-level/experiment/
  
  ./run_baseline.py -devices /dev/nvme0n1 /dev/nvme1n1 -trace_dirs $HEIMDALL/integration/client-level/data/*/*/*
  ```

+ **Output:** In directory `Heimdall/integration/client-level/data/*/*/*/nvme0n1...nvme1n1/baseline`
  
  1. Raw results containing each IO's latency.
  2. Aggregated statistics of this trace's replayed result.

---

### Train & Run Heimdall

+ Next, we train **Heimdall** by commands:

  + `flashnet` is `Heimdall`.
  + You can parallelize training *BUT NOT* replaying!
  + To maximize the CPU capability, check `top`

  ```bash
  cd $HEIMDALL/integration/client-level/experiment/
  
  ./run_flashnet.py -devices /dev/nvme0n1 /dev/nvme1n1 -trace_dirs $HEIMDALL/integration/client-level/data/*/*/* -only_training -resume
  ```

+ Training phase will take around one hour. After getting the training results, we replay workload using Heimdall by:

  ```bash
  cd $HEIMDALL/integration/client-level/experiment/
  
  ./run_flashnet.py -devices /dev/nvme0n1 /dev/nvme1n1 -trace_dirs $HEIMDALL/integration/client-level/data/*/*/* -only_replaying
  ```

+ **Output:** In directory `Heimdall/integration/client-level/data/*/*/*/nvme0n1...nvme1n1/flashnet`.

  + The results format is the same as baseline, including raw results of each IO's latency and aggregated statistics.

---

### Run Random

+ We continue to produce one of the baseline methods: **Random**. Random method submits IO to any one of the devices randomly to achieve load balancing. You can compile it by:

  ```bash
  cd $HEIMDALL/integration/client-level/experiment/random
  
  make
  ```

+ Then, replay traces using Random by:

  ```bash
  cd $HEIMDALL/integration/client-level/experiment/
  
  ./run_random.py -devices /dev/nvme0n1 /dev/nvme1n1 -trace_dirs $HEIMDALL/integration/client-level/data/*/*/*
  ```

+ **Output:** In directory `Heimdall/integration/client-level/data/*/*/*/nvme0n1...nvme1n1/random`.

---

### Run Hedging P95

+ **Hedging** as one of our baselines in evaluation, send a duplicate I/O after a p95 latency timeout to cut the tail latency. It is compile it by command:

  ```bash
  cd $HEIMDALL/integration/client-level/experiment/hedging
  
  make
  ```

+ Then, replay traces using Hedging-P95 by:

  ```bash
  cd $HEIMDALL/integration/client-level/experiment/
  
  ./run_hedging.py -devices /dev/nvme0n1 /dev/nvme1n1 -hedging_percentile 98 -trace_dirs $HEIMDALL/integration/client-level/data/*/*/*
  ```

+ **Output:** In directory `Heimdall/integration/client-level/data/*/*/*/nvme0n1...nvme1n1/hedging`.

---

### Run LinnOS Client

+ **LinnOS** is also an IO latencies reduction method powered by machine learning. Before replaying traces using LinnOS, it should also go through a training phase. You can train LinnOS by:

  ```bash
  cd $HEIMDALL/integration/client-level/experiment/
  
  ./run_linnos.py -devices /dev/nvme0n1 /dev/nvme1n1 -trace_dirs $HEIMDALL/integration/client-level/data/*/*/* -only_training
  ```

+ It may take around 1 hour to train LinnOS. After get the model trained, we can replay traces using Linnos by:

  ```bash
  cd $HEIMDALL/integration/client-level/experiment/
  
  ./run_linnos.py -devices /dev/nvme0n1 /dev/nvme1n1 -trace_dirs $HEIMDALL/integration/client-level/data/*/*/* -only_replaying
  ```

+ **Output:** In directory `Heimdall/integration/client-level/data/*/*/*/nvme0n1...nvme1n1/linnos`.

---

### Run LinnOS + Hedging

+ LinnOS can be married with Hedging to further cut the tail latency. You can run LinnOS + Hedging percentile 98 by:

  ```bash
  cd $HEIMDALL/integration/client-level/experiment/
  
  ./run_linnos_hedging.py -devices /dev/nvme0n1 /dev/nvme1n1 -trace_dirs $HEIMDALL/integration/client-level/data/*/*/* -hedging_percentile 98
  ```

+ **Output:** In directory `Heimdall/integration/client-level/data/*/*/*/nvme0n1...nvme1n1/linnos_hedging`.

---

### Latencies Characteristics Analysis

+ In the above steps, we have replayed traces with Heimdall and other baseline methods (Baseline, Random, Hedging P95, LinnOS, LinnOS + Hedging). Now, we would like to analyze and evaluate their performance in a aggregated and visualizable way. 

+ First, the IO latencies Cumulative distribution function (CDF) can be visualized by:

  ```bash
  cd $HEIMDALL/integration/client-level/trace_analysis
  
  ./analyze_trace_profile.py -trace_dirs $HEIMDALL/integration/client-level/data/*/*/*/*/
  ```

  + Output figures will be `cdf_all_algo.png`, which plots the IO latencies CDF of all methods (Heimdall and baseline methods) in a single graph.

+ Next, we are going to evaluate on another two important characteristics:

  1. **Average IO Latencies.**
  2. **Percentiles Latencies**: The IO latency values across thorough range of percentiles (p50, p80, p90, p95, p99, p99.9, p99.99).
  
  + Please generate the latency characteristics by:
  
    ```bash
    cd $HEIMDALL/integration/client-level/algo_analysis
    
    ./generate_latency_stats.py -trace_dirs $HEIMDALL/integration/client-level/data/*/*/*/*/
    ```
  
  Then, You can get evaluation figures about the above characteristics by running jupyter notebook `tail_improvement.ipynb`