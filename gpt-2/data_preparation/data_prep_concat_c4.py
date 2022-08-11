import glob
from itertools import chain

from datasets import load_dataset
from transformers import GPT2TokenizerFast, T5Tokenizer

# EC2 を FSx for Lustre へマウントし、 そのディレクトリを指定
# マウント方法参考 URL: https://docs.aws.amazon.com/ja_jp/fsx/latest/LustreGuide/mounting-ec2-instance.html
RAW_DATA_DIR = "../fsx/ns1/en_unzip"
SAVE_DIR = "../fsx/ns1/en_gpt_preprocessed_2048_small"
BLOCK_SIZE = 2048

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
    # HuggingFace datasets から データをダウンロード。 マウントしてある FSx for Lustre 上にキャッシュを保存。
    c4_subset = load_dataset('allenai/c4', data_files='multilingual/c4-ja.*.json.gz', cache_dir=RAW_DATA_DIR)
    del c4_subset

    # キャッシュ に保存されているファイルのファイル名を取得、ここでは試しに1ファイルのみを読み込んでいる。
    train_data_files = glob.glob(RAW_DATA_DIR+"/c4-train.00000-of-01024*")
    print(train_data_files)
    validation_data_files = glob.glob(RAW_DATA_DIR+"/c4-validation.00000-of-00008*")
    print(validation_data_files)

    # tokenizer を選択
    # tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt-1b")
    # tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer = T5Tokenizer.from_pretrained("megagonlabs/t5-base-japanese-web")
    
    for train_data_file in train_data_files:
        dataset = load_dataset('json', data_files=[train_data_file], cache_dir=RAW_DATA_DIR + "/cache")
        print(dataset)

        dataset = dataset.map(lambda e: tokenizer(e['text']), num_proc=96)
        columns = dataset['train'].column_names
        columns.remove('input_ids')
        columns.remove('attention_mask')
        dataset = dataset['train'].remove_columns(columns)
        print(dataset)

        dataset = dataset.map(group_texts, fn_kwargs={"block_size": BLOCK_SIZE}, batched=True, num_proc=96)
        print(dataset)
        
        # Remove samples that length is less than BLOCK_SIZE
        dataset = dataset.filter(lambda e: len(e['input_ids']) >= BLOCK_SIZE, num_proc=96)
        print(dataset)

        dataset = dataset.shuffle(seed=42)
        train_path = SAVE_DIR + "/train"
        save_path=f"{train_path}/train_dataset_2048_filtered_{train_data_file[-19:]}"
        dataset.to_json(save_path, orient="records", lines=True)

    for validation_data_file in validation_data_files:
        dataset = load_dataset('json', data_files=[validation_data_file], cache_dir="./data/cache")
        print(dataset)

        dataset = dataset.map(lambda e: tokenizer(e['text']), num_proc=96)
        dataset = dataset['train'].remove_columns(['text', 'timestamp'])
        print(dataset)

        dataset = dataset.map(group_texts, fn_kwargs={"block_size": BLOCK_SIZE}, batched=True, num_proc=96)
        print(dataset)
        
        # Remove samples that length is less than BLOCK_SIZE
        dataset = dataset.filter(lambda e: len(e['input_ids']) >= BLOCK_SIZE, num_proc=96)
        print(dataset)
        
        dataset = dataset.shuffle(seed=42)
        validation_path = SAVE_DIR + "/validation"
        save_path=f"{validation_path}/validation_dataset_2048_filtered_{validation_data_file[-19:]}"
        dataset.to_json(save_path, orient="records", lines=True)
