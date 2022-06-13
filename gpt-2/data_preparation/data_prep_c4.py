import glob

from datasets import load_dataset
from transformers import GPT2TokenizerFast

RAW_DATA_DIR = "./en"
SAVE_DIR = "./en_gpt_preprocessed"


if __name__ == "__main__":
    c4_subset = load_dataset('allenai/c4', data_files='en/*.json.gz', cache_dir=RAW_DATA_DIR)
    del c4_subset

    train_data_files = glob.glob(RAW_DATA_DIR+"/c4-train.*")
    validation_data_files = glob.glob(RAW_DATA_DIR+"/c4-validation.*")
    
    for train_data_file in train_data_files:
        dataset = load_dataset('json', data_files=[train_data_file], cache_dir=RAW_DATA_DIR + "/cache")
        print(dataset)

        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

        # Tokenize dataset and truncate the max length by 512
        dataset = dataset.map(lambda e: tokenizer(e['text'], max_length=512, truncation=True), num_proc=96)
        # Remove samples that length is less than 512
        dataset = dataset.filter(lambda e: len(e['input_ids']) >= 512, num_proc=96)
        print(dataset)
        dataset = dataset.remove_columns('text')
        dataset = dataset.shuffle(seed=42)
        
        train_path = SAVE_DIR + "/train"
        save_path=f"{train_path}/train_dataset_512_filtered_{train_data_file[9:]}"
        dataset.to_json(save_path, orient="records", lines=True)

    for validation_data_file in validation_data_files:
        dataset = load_dataset('json', data_files=[validation_data_file], cache_dir="./data/cache")
        print(dataset)

        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

        # Tokenize dataset and truncate the max length by 512
        dataset = dataset.map(lambda e: tokenizer(e['text'], max_length=512, truncation=True), num_proc=96)
        # Remove samples that length is less than 512
        dataset = dataset.filter(lambda e: len(e['input_ids']) >= 512, num_proc=96)
        print(dataset)
        dataset = dataset.remove_columns('text')
        
        validation_path = SAVE_DIR + "/validation"
        save_path=f"{validation_path}/validation_dataset_512_filtered_{validation_data_file[14:]}"
        dataset.to_json(save_path, orient="records", lines=True)    
