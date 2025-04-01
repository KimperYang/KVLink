"""
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --config_file configs/h100_config.yaml \
    --main_process_port 25678 block_attn_trainer.py

CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/single_gpu.yaml \
    --main_process_port 25678 block_attn_trainer.py

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --config_file configs/fsdp.yaml \
    --main_process_port 25678 block_attn_trainer.py
"""
from typing import Tuple

import datasets
from transformers import Qwen2ForCausalLM, AutoTokenizer, TrainingArguments

from src.data.titan_data_utils import Qwen_SumAttentionPreprocessor

def load_from_disk_then_process(
    data_component_name: str,
    preprocessor: Qwen_SumAttentionPreprocessor,
) -> Tuple[datasets.IterableDataset, datasets.Dataset]:
    """
    load the downloaded data from disk and then pair it with the preprocessor
    """
    if data_component_name in ["text", "text_mem", "text_inst"]:
        data_path = f"dataset_cache/processed/fineweb/{data_component_name}"
        if data_component_name == "text":
            preprocessor_fn = preprocessor.process_text
        else:
            raise NotImplementedError()
        remove_columns = [
            "text", "id", "dump", "url", "date",
            "file_path", "language", "language_score", "token_count",
        ]
        num_shards = 512
    elif data_component_name in ["sft_mem"]:
        data_path = f"dataset_cache/processed/daringanteater/{data_component_name}"
        if data_component_name == "sft_mem":
            preprocessor_fn = preprocessor.process_sftmem
        else:
            raise NotImplementedError()
        remove_columns=["system", "mask", "dataset", "conversations"]
        num_shards = 32
    elif data_component_name in ["tulu"]:
        data_path = "dataset_cache/processed/tulu/sft"
        if data_component_name == "tulu":
            preprocessor_fn = preprocessor.process_tulu
        else:
            raise NotImplementedError()
        remove_columns=["id", "messages", "source"]
        num_shards = 32
    elif data_component_name in ["qa", "qa_mem"]:
        data_path = f"dataset_cache/processed/block_qa/{data_component_name}"
        if data_component_name == "qa":
            preprocessor_fn = preprocessor.process_qa
        elif data_component_name == "qa_mem":
            preprocessor_fn = preprocessor.process_qamem
        else:
            raise NotImplementedError()
        remove_columns=['prompt', 'question', 'answers', 'generated', 'inputs', 'documents']
        num_shards = 32
    elif data_component_name in ["xsum"]:
        data_path = f"dataset_cache/processed/xsum/{data_component_name}"
        preprocessor_fn = preprocessor.process_xsum
        remove_columns=['document', 'summary', 'id']
        num_shards = 32
    else:
        raise NotImplementedError()
    data_component: datasets.DatasetDict = datasets.load_from_disk(data_path)
    # print(data_component.cleanup_cache_files())

    streaming_train_dataset = data_component["train"]
    # streaming_train_dataset = data_component["train"]
    training_data = streaming_train_dataset.map(
        preprocessor_fn,
        remove_columns=remove_columns,
        num_proc=64,
        batched=False,
    )

    eval_dataset = data_component["test"]
    eval_data = eval_dataset.map(
        preprocessor_fn,
        remove_columns=remove_columns,
        num_proc=64,
        batched=False,
        # load_from_cache_file=False
    )

    return training_data, eval_data


def main():

    tokenizer_path = "Qwen/Qwen2.5-14B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    special_token_start = len(tokenizer)
    max_memory_num = 40
    new_special_tokens = [f"<link_{i}>" for i in range(max_memory_num * 5)] + ["<mem_start>", "<mem_end>"]
    special_tokens_dict = {"additional_special_tokens": new_special_tokens}

    tokenizer.add_special_tokens(special_tokens_dict, replace_additional_special_tokens=False)

    mem_start = len(tokenizer) - 2
    mem_end = len(tokenizer) - 1

    preprocessor = Qwen_SumAttentionPreprocessor(
        tokenizer=tokenizer,
        max_len=4096,
        special_token_start=special_token_start,
        mem_start=mem_start,
        mem_end=mem_end,
        reencode_num=5,
        max_memory_num= max_memory_num,
    )

    load_from_disk_then_process("text", preprocessor)
    load_from_disk_then_process("tulu", preprocessor)
    load_from_disk_then_process("qa", preprocessor)
    load_from_disk_then_process("qa_mem", preprocessor)
    load_from_disk_then_process("xsum", preprocessor)
    load_from_disk_then_process("sft_mem", preprocessor)

if __name__ == "__main__":
    main()
