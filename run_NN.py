
import argparse
import sys
import re
import torch
from torch.utils.data import random_split

from CNN_train import WaveformCNN, Trainer as CNN_Trainer, create_dataset
from Transformer_train import WaveformTransformer, Trainer as Transformer_Trainer

def main():
    parser = argparse.ArgumentParser(description="Train NN Model for Nanopore Data")
    parser.add_argument('--model', type=str, choices=['cnn', 'transformer', 'CNN', 'Transformer'], required=True, 
                        help="使用するモデル ('cnn' または 'transformer')")
    parser.add_argument('--ca', type=str, required=True, help="学習用 CA15-3 のデータフォルダパス")
    parser.add_argument('--cea', type=str, required=True, help="学習用 CEA のデータフォルダパス")
    parser.add_argument('--epochs', type=int, default=10, help="学習のエポック数 (デフォルト: 10)")
    parser.add_argument('--save_path', type=str, default=None, help="保存するモデルのファイル名")

    # ==========================================
    # コマンドの全角スペース対策 ＆ ハイフン省略の処理
    # ==========================================
    # 入力された引数を一旦すべてつなげる
    raw_args_str = " ".join(sys.argv[1:])
    # 全角スペース・半角スペースなどで綺麗に文字をバラバラに分割する
    split_args = re.split(r'[　 \t]+', raw_args_str)

    valid_keys =['model', 'ca', 'cea', 'epochs', 'save_path']
    processed_args =[]
    for arg in split_args:
        if not arg: continue # 空文字は無視
        # キーワードがあれば自動で "--" をつける
        if arg in valid_keys:
            processed_args.append(f'--{arg}')
        else:
            processed_args.append(arg)

    args = parser.parse_args(processed_args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using Device: {device}")

    # ==========================================
    # データセットの準備 (8:2でTrain/Testに分割)
    # ==========================================
    print("Loading dataset...")
    dataset = create_dataset([
        (args.ca, 0),
        (args.cea, 1)
    ])
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset,[train_size, test_size])

    # ==========================================
    # モデルの準備
    # ==========================================
    model_type = args.model.lower()
    if model_type == "cnn":
        model = WaveformCNN()
        trainer = CNN_Trainer(model, device, lr=0.001)
        save_path = args.save_path if args.save_path else "cnn_model.pth"
    elif model_type == "transformer":
        model = WaveformTransformer()
        trainer = Transformer_Trainer(model, device, lr=0.0001)
        save_path = args.save_path if args.save_path else "transformer_model.pth"

    # ==========================================
    # 学習の実行と保存
    # ==========================================
    print(f"\n--- Start training {model_type.upper()} ---")
    trainer.run(train_dataset, num_epochs=args.epochs, save_path=save_path)
    print(f"Model saved to: {save_path}")

    # 学習がうまく行ったかの確認として、20%のテストデータで評価
    auc = trainer.evaluate(test_dataset)
    print(f"Validation AUC (20% split): {auc:.4f}")

if __name__ == "__main__":
    main()