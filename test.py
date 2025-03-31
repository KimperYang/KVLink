import argparse
import json
import random
import sys
from transformers import AutoTokenizer
# If your Qwen_SumAttentionPreprocessor code is in qwen_preprocessor.py,
# adjust this import statement accordingly:
from src.data.titan_data_utils import Qwen_SumAttentionPreprocessor

# If you use a Hugging Face tokenizer:
# from transformers import AutoTokenizer

def test_length_consistency(data, preprocessor, process_func_name, max_examples=100):
    """
    Iterates through 'data' and calls preprocessor.<process_func_name>(example).
    Checks whether 'input_ids', 'labels', 'segment_ids' match in length.
    """
    process_func = getattr(preprocessor, process_func_name, None)
    if process_func is None:
        print(f"[WARNING] No function '{process_func_name}' found in preprocessor.")
        return

    print(f"--- Testing function: {process_func_name} ---")
    mismatch_count = 0

    # If dataset is huge, sample up to max_examples to speed up the test
    if len(data) > max_examples:
        data_indices = random.sample(range(len(data)), max_examples)
    else:
        data_indices = range(len(data))

    for i in data_indices:
        example = data[i]
        try:
            output = process_func(example)
            i_ids = output["input_ids"]
            labels = output["labels"]
            seg_ids = output["segment_ids"]
        except Exception as e:
            print(f"[ERROR] Exception while processing example {i} with {process_func_name}: {e}")
            mismatch_count += 1
            continue

        if not (len(i_ids) == len(labels) == len(seg_ids)):
            print(
                f"[MISMATCH] Example {i} | "
                f"input_ids={len(i_ids)}, labels={len(labels)}, segment_ids={len(seg_ids)}"
            )
            mismatch_count += 1

    if mismatch_count == 0:
        print(f"[OK] No length mismatches found for {process_func_name}.")
    else:
        print(f"[DONE] Found {mismatch_count} mismatch(es) in {process_func_name}.")

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--data_path",
    #     type=str,
    #     required=True,
    #     help="Path to the JSON dataset file."
    # )
    # Add any needed arguments for your preprocessor instantiation:
    # e.g., parser.add_argument("--max_len", type=int, default=2048)
    # parser.add_argument("--special_token_start", type=int, default=32000)
    # etc.

    # args = parser.parse_args()

    # 1) Load the data from JSON
    with open("dataset_cache/processed/block_qa/qa", "r", encoding="utf-8") as f:
        data = json.load(f)

    # 2) Instantiate your tokenizer (if you have a custom one or HF’s)
    # tokenizer = AutoTokenizer.from_pretrained("some-tokenizer")
    # Or your LLaMA32Tokenizer etc.

    # 3) Create the preprocessor instance with the required arguments
    preprocessor = Qwen_SumAttentionPreprocessor(
        tokenizer=AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct"),             # e.g. your tokenizer
        max_len=4096,               # fill in your desired max length
        special_token_start=128011,   # fill in the special start int
        mem_start=128011,             # ...
        mem_end=128001,               # ...
        reencode_num=5,          # ...
        max_memory_num=40,        # ...
        qa_document_num=10        # ...
    )

    # 4) Test each process function for length consistency
    # test_length_consistency(data, preprocessor, "process_sftmem")
    # test_length_consistency(data, preprocessor, "process_sft")
    # test_length_consistency(data, preprocessor, "process_text")
    test_length_consistency(data, preprocessor, "process_qamem")
    test_length_consistency(data, preprocessor, "process_qa")
    # test_length_consistency(data, preprocessor, "process_tulu")
    # test_length_consistency(data, preprocessor, "process_xsum")

if __name__ == "__main__":
    main()
