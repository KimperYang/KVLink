## KVLink: Accelerating LLMs via Efficient KV Cache Reuse

This is the official implementation for the paper, "KVLink: Accelerating LLMs via Efficient KV Cache Reuse".

### Preparation

#### 1. Virtural Environment

The training code is built upon [torchtitan](https://github.com/pytorch/torchtitan) and torchtune (https://github.com/pytorch/torchtune) for efficient gradient checkpointing and parallism.

1. install the preview version of PyTorch:

```
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
```

2. install the other required dependencies:

```
pip install -r requirements.txt
```

3. install torchtune:

```
pip install torchtune
pip install -e ./
```

#### 2. Data Preprocessing

We provide the code for data preprocessing under `scripts/data_process/`. The running commands and documents are availabel in the header of each file. You can preprocess the training data using:

```
python scripts/data_process/fineweb.py --num_samples=10000000 --min_length_for_memory=2048 --validation_size=3000
python scripts/data_process/daring_anteater.py --max_length=4096 --validation_size=2000
python scripts/data_process/tulu.py --max_length=4096 --validation_size=2000
python scripts/data_process/sum.py --max_length=4096 --validation_size=1000
```

For preprocessing the QA data, you can first access the original 2WikiMultiHopQA data using

```
git lfs install 
git clone https://huggingface.co/datasets/xanhho/2WikiMultihopQA
```

and the TriviaQA data using

```
git clone https://github.com/facebookresearch/FiD
bash FiD/get-data.sh 
```

After that, run the following script to retrieve the relevant QA document using Contriever, and generate answers using GPT4

```
python scripts/data_process/gpt_answer.py
```

Lastly, run the preprocessing script as other datasets

```
python scripts/data_process/block_qa.py --max_length=4096 --validation_size=2000
```

### Implementation

The `cross-document reconnection with summary tokens` is impmented in `src/data/titan_preprocess.py`, Line 498, where the preprocessor will add summary tokens to each context chunk. The special attention mask (Figure 2 of the paper) is implemented in the function `make_segment_mask`, located in Line 1069 of `src/data/titan_preprocess.py`.





### Model Training

Our trainer is mainly based on torchtitan (see `titan_trainer_kvlink.py`). All configurations are also listed in the same file. For example, the config name `data_original_step6k_bsz64_link_5_selective_ckpt` corresponds to the following training:

- original data mixture (data mixture is defined in `src/training/titan_training_utils.py`, DATASET_MAPPING)
- training steps 6000
- batch size 64
- KVLink 5
- apply selective gradient checkpinting to save memory
- learning rate 5e-6

Before you train the model, first download the tokenizer and the model:

1. Download the tokenizer

```
python src/data/titan_download_tokenizer.py \
	--repo_id meta-llama/Llama-3.2-1B-Instruct \
	--tokenizer_path "original" \
	--local_dir data/titan_tokenizer/ \
	--hf_token=YOUR_HF_TOKEN
```



2. Download the model

```
tune download meta-llama/Llama-3.2-1B-Instruct \
    --output-dir model_cache/Llama-3.2-1B-Instruct \
    --ignore-patterns "original/consolidated.00.pth" \
    --hf-token YOUR_HF_TOKEN \
```



Now you can run the training using the following script:

```
LOG_RANK=${LOG_RANK:-0}
NGPU=${NGPU:-"8"}

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
torchrun --nproc_per_node=8 --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
    --local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
    titan_trainer.py --config_name datav6_step6k_bsz64_reencode_5_selective_ckpt
```



By default, we use tensorboard to visualize the training log, which will be saved in `{job_dump_folder}` define in the training config.



### Evaluation

The evaluation code is availabel in `scripts/evaluation`

You need to convert the saved model checkpoint (in `DCP` format) to pytorch:

```
python -m torch.distributed.checkpoint.format_utils dcp_to_torch \
    torchtitan/outputs/checkpoint/step-1000 checkpoint.pt
```

Then run the evaluation using,

```
python scripts/evaluation/wiki_eval.py \
    --ckpt_path checkpoint.pt \
    --batch_size 10 \
    --reencode_num 5 \
    --attn_type "blocked" \
```

Note, for the evaluation of NQ, one extra 'pos' argument is needed (from 0 to 9) to specify the golden document index.

```
python scripts/evaluation/nq_eval.py \
    --ckpt_path checkpoint.pt \
    --batch_size 10 \
    --pos 0 \
    --reencode_num 5 \
    --attn_type "blocked" \
```
