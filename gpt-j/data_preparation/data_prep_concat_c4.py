from itertools import chain

from datasets import load_dataset
from transformers import GPT2TokenizerFast, T5Tokenizer

# max_context_width に合わせて、結合する長さのサイズを指定。今回は GTP-J に合わせて 2048 とした。
BLOCK_SIZE = 2048
# 開発用 EC2 を FSx for Lustre へマウントし、 そのディレクトリを指定
# マウント方法参考 URL: https://docs.aws.amazon.com/ja_jp/fsx/latest/LustreGuide/mounting-ec2-instance.html
RAW_DATA_DIR = "<fsx_mount_dir>/ns1/ja_raw_data"
SAVE_DIR = f"<fsx_mount_dir>/ns1/ja_megagon_preprocessed_{BLOCK_SIZE}"

# データ前処理用の設定
# 処理する学習データのファイル数
JSON_NUM_TRAIN = 1024
# 処理する評価データのファイル数
JSON_NUM_VAL = 8
# 並列処理実行時のプロセスの数
NUM_PROC = 96

def group_texts(examples, block_size):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size

    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    return result


if __name__ == "__main__":
    # tokenizer を選択
    # tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt-1b")
    # tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer = T5Tokenizer.from_pretrained("megagonlabs/t5-base-japanese-web")
    
    for i in range(JSON_NUM_TRAIN):
        train_data_file ='multilingual/c4-ja.tfrecord-%05d-of-01024.json.gz' % (i)
        dataset = load_dataset('allenai/c4', data_files=train_data_file, cache_dir=RAW_DATA_DIR)
        print(dataset)

        dataset = dataset.map(lambda e: tokenizer(e['text']), num_proc=NUM_PROC)
        columns = dataset['train'].column_names
        columns.remove('input_ids')
        columns.remove('attention_mask')
        dataset = dataset['train'].remove_columns(columns)
        print(dataset)

        dataset = dataset.map(group_texts, fn_kwargs={"block_size": BLOCK_SIZE}, batched=True, num_proc=NUM_PROC)
        print(dataset)
        
        # Remove samples that length is less than BLOCK_SIZE
        dataset = dataset.filter(lambda e: len(e['input_ids']) >= BLOCK_SIZE, num_proc=NUM_PROC)
        print(dataset)

        dataset = dataset.shuffle(seed=42)
        train_path = SAVE_DIR + "/train"
        save_path=f"{train_path}/train_dataset_2048_filtered_{train_data_file[-22:-3]}"
        dataset.to_json(save_path, orient="records", lines=True, num_proc=4)

    #for validation_data_file in validation_data_files:
    for i in range(JSON_NUM_VAL):
        validation_data_file ='multilingual/c4-ja-validation.tfrecord-%05d-of-00008.json.gz' % (i)
        dataset = load_dataset('allenai/c4', data_files=validation_data_file, cache_dir=RAW_DATA_DIR)
        print(dataset)

        dataset = dataset.map(lambda e: tokenizer(e['text']), num_proc=NUM_PROC)
        dataset = dataset['train'].remove_columns(['text', 'timestamp'])
        print(dataset)

        dataset = dataset.map(group_texts, fn_kwargs={"block_size": BLOCK_SIZE}, batched=True, num_proc=NUM_PROC)
        print(dataset)
        
        # Remove samples that length is less than BLOCK_SIZE
        dataset = dataset.filter(lambda e: len(e['input_ids']) >= BLOCK_SIZE, num_proc=NUM_PROC)
        print(dataset)
        
        dataset = dataset.shuffle(seed=42)
        validation_path = SAVE_DIR + "/validation"
        save_path=f"{validation_path}/validation_dataset_2048_filtered_{validation_data_file[-22:-3]}"
        dataset.to_json(save_path, orient="records", lines=True, num_proc=4)
