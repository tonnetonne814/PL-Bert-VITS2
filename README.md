# Phoneme-Level BERT-VITS2 (48000Hz 日本語版)

このリポジトリは、 48000Hzの日本語音声を学習および出力できるように編集した[VITS2](https://github.com/daniilrobnikov/vits2)に、
[Phoneme-Level Japanese BERT](https://github.com/yl4579/PL-BERT)の中間潜在表現を用いた音声合成モデルです。

## 1. 環境構築

Anacondaによる実行環境構築を想定する。

0. Anacondaで"PLBERTVITS2"という名前の仮想環境を作成する。[y]or nを聞かれたら[y]を入力する。
    ```sh
    conda create -n PLBERTVITS2 python=3.8    
    ```
0. 仮想環境を有効化する。
    ```sh
    conda activate PLBERTVITS2
    ```
0. このレポジトリをクローンする（もしくはDownload Zipでダウンロードする）

    ```sh
    git clone https://github.com/tonnetonne814/PL-Bert-VITS2.git
    cd PL-Bert-VITS2 # フォルダへ移動
    ```

0. [https://pytorch.org/](https://pytorch.org/)のURLよりPyTorchをインストールする。
    
    ```sh
    pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 # cuda11.7 linuxの例
    ```

0. その他、必要なパッケージをインストールする。
    ```sh
    pip install -r requirements.txt 
    ```
0. Monotonoic Alignment Searchをビルドする。
    ```sh
    cd monotonic_align
    mkdir monotonic_align
    python setup.py build_ext --inplace
    cd ..
    ```
1. [PL-BERT-ja](https://github.com/kyamauchi1023/PL-BERT-ja?tab=readme-ov-file)より、日本語版のPhoneme−Level Bertの事前学習モデルをダウンロード及び展開する。

## 2. データセットの準備

[JVNV Speech dataset](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvnv_corpus?authuser=0)による48000Hz音声の学習生成を想定する。

1. [JVNV Speech dataset](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvnv_corpus?authuser=0)をダウンロード及び展開する。

1. 展開したフォルダの中にあるjvnv_ver1フォルダ及びplb-ja_10000000-stepsフォルダを指定して、以下を実行する。
    ```sh
    python3 ./preprocess_ja.py --jvnv_dir ./path/to/jvnv_ver1/ --pl_bert_dir ./path/to/plb-ja_10000000-steps
    ```

    
## 3. [configs](configs)フォルダ内のjsonを編集
主要なパラメータを説明します。必要であれば編集する。
| 分類  | パラメータ名      | 説明                                                      |
|:-----:|:-----------------:|:---------------------------------------------------------:|
| train | log_interval      | 指定ステップ毎にロスを算出し記録する                      |
| train | eval_interval     | 指定ステップ毎にモデル評価を行う                          |
| train | epochs            | 学習データ全体を学習する回数                          |
| train | batch_size        | 一度のパラメータ更新に使用する学習データ数                |
| data  | training_files    | 学習用filelistのテキストパス                              |
| data  | validation_files  | 検証用filelistのテキストパス                              |


## 4. 学習
次のコマンドを入力することで、学習を開始する。
> ⚠CUDA Out of Memoryのエラーが出た場合には、config.jsonにてbatch_sizeを小さくする。

```sh
python train_ms.py --config configs/jvnv_base.json -m JVNV_Dataset
```


学習経過はターミナルにも表示されるが、tensorboardを用いて確認することで、生成音声の視聴や、スペクトログラム、各ロス遷移を目視で確認することができます。
```sh
tensorboard --logdir logs
```

## 5. 推論
次のコマンドを入力することで、推論を開始する。config.jsonへのパスと、生成器モデルパスと、PL-BERT-jaのフォルダを指定する。
```sh
python3 inference.py --model_ckpt_path ./path/to/ckpt.pth --model_cnfg_path ./path/to/config.json --pl_bert_dir /path/to/plb-ja_10000000-steps
```
Terminal上にて使用するデバイスを選択後、テキストを入力することで、音声が生成さされます。音声は自動的に再生され、infer_logsフォルダ（存在しない場合は自動作成）に保存されます。

## 事前学習モデル
- 後ほど追加します。


## 参考文献
- https://github.com/fishaudio/Bert-VITS2
- https://github.com/yl4579/PL-BERT
- https://github.com/daniilrobnikov/vits2
- https://github.com/kyamauchi1023/PL-BERT-ja
