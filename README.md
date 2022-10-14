# WWW23_Temporary

#### We use this temporay repo to provide supplementary material for our manuscript "Pretrained Embedding for E-commerce Machine Learning: When it Fails and Why?".

#### In "Online material.pdf", we provide the proofs to the theoretical claims we made in the paper. All the contents are properly anonymized, and please do not distribute. 

## Running code:

We created three scripts for directly running our experiments for the Instacart, MovieLens-1M and Amazon Electronics dataset. 

Due to the enumeration of different CL settings for window size and \#negative samples, they will consume quite some time. The instructions for the data dir setup are provided in **data/README.md**. 

```bash
python instacart_run.py
python ml-1m-run.py
python amzn-run.py
```

The arguments are as follow:
```bash
  --log_dir: the logging directory for saving the results
  --dat_dir: the data directory, e.g. data/ml-1m/
  --emb_dim: the dimension of the embeddings. We use 32 in our experiments unless specificed
  --reload: whether to reload the environment data from a previous implementation, or start fresh
  --save_emb: whether to save the pre-trained embeddings under each CL setting
  --GPU: specify the GPU usage.
```

For the sequential recommendation tasks, we implement the two-tower Dense4Rec, GRU4Rec and Attn4Rec, with the models' source code in **src/model.py**. 

We follow the standard data processing steps in reco sys, and the implementations can be found in **src/data.py**.





