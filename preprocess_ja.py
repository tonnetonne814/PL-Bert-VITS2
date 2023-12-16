import argparse
import os 
import polars
import random
from PL_BERT_ja.text_utils import TextCleaner
from PL_BERT_ja.phonemize import phonemize
import torch
from tqdm import tqdm

from PL_BERT_ja.model import MultiTaskModel
from transformers import AlbertConfig, AlbertModel
from transformers import BertJapaneseTokenizer
import yaml, torch

def preprocess(dataset_dir, pl_bert_dir):

    n_val_test_file = 10
    filelist_dir = "./filelists/"
    dataset_name = "jvnv_ver1"
    os.makedirs(filelist_dir, exist_ok=True)
    split_symbol = "||||"

    transcript_csv_df = polars.read_csv(os.path.join(dataset_dir, "jvnv_v1", "transcription.csv"),has_header=False)[:, 0]
    emo_list = os.listdir(os.path.join(dataset_dir,"jvnv_v1", "F1"))
    style_list = os.listdir(os.path.join(dataset_dir,"jvnv_v1", "F1", "anger"))
    
    pl_bert_savedir = "./pl_bert_embeddings"
    os.makedirs(pl_bert_savedir, exist_ok=True)
    pl_bert_model, pl_bert_config, device = get_pl_bert_ja(dir=pl_bert_dir)
    pl_bert_cleaner = TextCleaner()
    pl_bert_tokenizer = BertJapaneseTokenizer.from_pretrained(pl_bert_config['dataset_params']['tokenizer'])

    hidden_size = pl_bert_config["model_params"]["hidden_size"]
    n_layers = pl_bert_config["model_params"]["num_hidden_layers"] + 1

    filelists = list() 
    spk_g = ["F", "M"]
    for line in tqdm(transcript_csv_df):
        index_name, emo_prefix, text = line.split("|")
        emotion, style, file_idx = index_name.split("_")
        text = text.replace("\n", "")

        phonemes = ''.join(phonemize(text,pl_bert_tokenizer)["phonemes"])
        input_ids = pl_bert_cleaner(phonemes)
        with torch.inference_mode():
            hidden_stats  = pl_bert_model(torch.tensor(input_ids, dtype=torch.int64, device=device).unsqueeze(0))[-1]["hidden_states"]
        save_tensor = torch.zeros(size=(n_layers, len(input_ids), hidden_size), device=device)
        for idx, hidden_stat in enumerate(hidden_stats):
            save_tensor[idx, :, :] = hidden_stat
        torch.save(save_tensor.to('cpu').detach(), os.path.join(pl_bert_savedir, f"{index_name}.PlBertJa"))

        for g_idx in range(2):
            for spk_idx in range(2):
                spk_ID = str(g_idx + spk_idx*2)
                spk = spk_g[g_idx] + str(spk_idx+1)
                wav_path = os.path.join(dataset_dir, "jvnv_v1", spk, emotion, style, f"{spk}_{emotion}_{style}_{file_idx}.wav")
                filelists.append(f"{wav_path}{split_symbol}{spk_ID}{split_symbol}{phonemes}{split_symbol}{text}{split_symbol}{index_name}{split_symbol}emo:{str(emo_list.index(emotion))}{split_symbol}style:{str(style_list.index(style))}\n")

    val_list = list()
    test_list = list()
    for idx in range(n_val_test_file*2):
        target_idx = random.randint(0, len(filelists))
        target_line = filelists.pop(target_idx)
        if idx % 2 == 1:
            val_list.append(target_line)
        else:
            test_list.append(target_line)

    write_txt(filelists, os.path.join(filelist_dir, f"{dataset_name}_train.txt"))
    write_txt(val_list, os.path.join(filelist_dir, f"{dataset_name}_val.txt"))
    write_txt(test_list, os.path.join(filelist_dir, f"{dataset_name}_test.txt"))
    
    return 0

def write_txt(lists, path):
    with open(path, mode="w", encoding="utf-8") as f:
        f.writelines(lists)

def get_pl_bert_ja(dir):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_path=os.path.join(dir, "config.yml")
    config = yaml.safe_load(open(config_path))

    albert_base_configuration = AlbertConfig(**config['model_params'])
    bert_ = AlbertModel(albert_base_configuration).to(device)
    #num_vocab = max([m['token'] for m in token_maps.values()]) + 1  # 30923 + 1
    bert = MultiTaskModel(
        bert_,
        num_vocab=30923 + 1,
        num_tokens=config['model_params']['vocab_size'],
        hidden_size=config['model_params']['hidden_size']
    )

    model_ckpt_path = os.path.join(dir,"10000000.pth.tar")
    checkpoint = torch.load(model_ckpt_path)
    bert.load_state_dict(checkpoint['model'], strict=False)
    
    bert.to(device)
    return bert, config, device
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jvnv_dir", default="./jvnv_ver1/")
    parser.add_argument("--pl_bert_dir", default="./plb-ja_10000000-steps/")

    args = parser.parse_args()

    preprocess(args.jvnv_dir, args.pl_bert_dir)