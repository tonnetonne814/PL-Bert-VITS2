
from models import SynthesizerTrn
import argparse
import utils
from PL_BERT_ja.text.symbols import symbols
import json
from preprocess_ja import get_pl_bert_ja
import torch
import soundcard as sc
import time
import os
import soundfile as sf
from transformers import BertJapaneseTokenizer
import torch
from PL_BERT_ja.text_utils import TextCleaner
from PL_BERT_ja.phonemize import phonemize
import commons
from text import cleaned_text_to_sequence, text_to_sequence

def inference(model_ckpt_path, model_config_path, pl_bert_dir, is_save=True):
    with open(model_config_path, "r") as f:
            data = f.read()
    config = json.loads(data)
    hps = utils.HParams(**config)

    if hps.model.use_noise_scaled_mas is True :
        print("Using noise scaled MAS for VITS2")
        use_noise_scaled_mas = True
        mas_noise_scale_initial = 0.01
        noise_scale_delta = 2e-6

    net_g = SynthesizerTrn(
        len(symbols)+1,
        hps.data.n_mel_channels,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        mas_noise_scale_initial=mas_noise_scale_initial,
        noise_scale_delta=noise_scale_delta,
        **hps.model,
    )

    pl_bert_model, pl_bert_config, device = get_pl_bert_ja(dir=pl_bert_dir)
    pl_bert_cleaner = TextCleaner()
    pl_bert_tokenizer = BertJapaneseTokenizer.from_pretrained(pl_bert_config['dataset_params']['tokenizer'])

    net_g, _, _, _ = utils.load_checkpoint( model_ckpt_path, net_g, optimizer=None)

    # play audio by system default
    speaker = sc.get_speaker(sc.default_speaker().name)

    # parameter settings
    noise_scale     = torch.tensor(0.66)    # adjust z_p noise
    noise_scale_w   = torch.tensor(0.8)   # adjust SDP noise
    length_scale    = torch.tensor(1.0)     # adjust sound length scale (talk speed)

    if is_save is True:
        n_save = 0
        save_dir = os.path.join("./infer_logs/")
        os.makedirs(save_dir, exist_ok=True)

    net_g = net_g.to(device)
    pl_bert_model = pl_bert_model.to(device)

    ### Dummy Input ###
    with torch.inference_mode():
        dummy_text = "色々疲れちまったけど、やっぱ音声合成してるときが一番ワクワクするんだよな。"

        # get bert features
        bert_features, phonemes = get_bert_features(dummy_text, pl_bert_model, pl_bert_tokenizer, pl_bert_config, pl_bert_cleaner, device, add_blank=hps.data.add_blank)
        x = get_text_ids(phonemes=phonemes,  
                         add_blank=hps.data.add_blank)
        x = x.unsqueeze(0)
        bert_features = bert_features.unsqueeze(0)
        x_lengths = torch.LongTensor([x.size(1)])
        sid =  torch.LongTensor([0])
        net_g.infer(x               .to(device), 
                    x_lengths       .to(device), 
                    bert_features   .to(device),
                    x_lengths       .to(device), 
                    sid             .to(device),
                    noise_scale=noise_scale.to(device), 
                    noise_scale_w=noise_scale_w.to(device), 
                    length_scale=length_scale.to(device),
                    max_len=1000)
        
    while True:
        # get text
        text = input("Enter text. ==> ")
        if text=="":
            print("Empty input is detected... Exit...")
            break
        
        # measure the execution time 
        torch.cuda.synchronize()
        start = time.time()

        # required_grad is False
        with torch.inference_mode():
            bert_features, phonemes = get_bert_features(text, pl_bert_model, pl_bert_tokenizer, pl_bert_config, pl_bert_cleaner, device, add_blank=hps.data.add_blank)
            x = get_text_ids(phonemes=phonemes,  
                             add_blank=hps.data.add_blank).unsqueeze(0)
            bert_features = bert_features.unsqueeze(0)
            x_lengths = torch.LongTensor([x.size(1)])
            sid =  torch.LongTensor([0])
            y_hat, _, _, _ = net_g.infer(x               .to(device), 
                                         x_lengths       .to(device), 
                                         bert_features   .to(device),
                                         x_lengths       .to(device), 
                                         sid             .to(device),
                                         noise_scale=noise_scale.to(device), 
                                         noise_scale_w=noise_scale_w.to(device), 
                                         length_scale=length_scale.to(device),
                                         max_len=1000)
            y_hat = y_hat.permute(0,2,1)[0, :, :].cpu().float().numpy().copy()

        # measure the execution time 
        torch.cuda.synchronize()
        elapsed_time = time.time() - start
        print(f"Gen Time : {elapsed_time}")
        
        # play audio
        speaker.play(y_hat, hps.data.sampling_rate)
        
        # save audio
        if is_save is True:
            n_save += 1
            data = y_hat
            try:
                save_path = os.path.join(save_dir, str(n_save).zfill(3)+f"_{text}.wav")
                sf.write(
                     file=save_path,
                     data=data,
                     samplerate=hps.data.sampling_rate,
                     format="WAV")
            except:
                save_path = os.path.join(save_dir, str(n_save).zfill(3)+f"_{text[:10]}〜.wav")
                sf.write(
                     file=save_path,
                     data=data,
                     samplerate=hps.data.sampling_rate,
                     format="WAV")

            print(f"Audio is saved at : {save_path}")

    return 0

def get_text_ids(phonemes, add_blank):

    text_norm = cleaned_text_to_sequence(phonemes)

    if add_blank:
        text_norm = commons.intersperse(text_norm, 0)

    text_norm = torch.LongTensor(text_norm)
    return text_norm


def get_bert_features(text, pl_bert_model, pl_bert_tokenizer, pl_bert_config, pl_bert_cleaner, device, add_blank):
    text = text.replace("\n", "")
    hidden_size = pl_bert_config["model_params"]["hidden_size"]
    n_layers = pl_bert_config["model_params"]["num_hidden_layers"] + 1
    phonemes = ''.join(phonemize(text,pl_bert_tokenizer)["phonemes"])
    input_ids = pl_bert_cleaner(phonemes)
    with torch.inference_mode():
        hidden_stats  = pl_bert_model(torch.tensor(input_ids, dtype=torch.int64, device=device).unsqueeze(0))[-1]["hidden_states"]
    save_tensor = torch.zeros(size=(n_layers, len(input_ids), hidden_size))
    for idx, hidden_stat in enumerate(hidden_stats):
        save_tensor[idx, :, :] = hidden_stat


    if add_blank is True:
        L, T, H = save_tensor.shape
        new_data = torch.zeros(size=(L,2*T+1,H), dtype=save_tensor.dtype)
        for idx in range(T):
            target_idx = idx*2+1
            new_data[:, target_idx, :] = save_tensor[:, idx, :]
        save_tensor =  new_data
        
    return save_tensor, phonemes

def text2input_ids():
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_ckpt_path", default="./logs/AddBlankTrue/G_54000.pth")
    parser.add_argument("--model_cnfg_path", default="./logs/AddBlankTrue/config.json")
    parser.add_argument("--pl_bert_dir",    default="./plb-ja_10000000-steps/")
    parser.add_argument("--is_save",        default=False)

    args = parser.parse_args()

    inference(args.model_ckpt_path, args.model_cnfg_path, args.pl_bert_dir, args.is_save)