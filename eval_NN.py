import argparse
import sys
import re
import torch

from CNN_train import WaveformCNN, Trainer as CNN_Trainer, create_dataset
from Transformer_train import WaveformTransformer, Trainer as Transformer_Trainer

def main():
    parser = argparse.ArgumentParser(description="Evaluate Trained NN Model")
    parser.add_argument('--model', type=str, choices=['cnn', 'transformer', 'CNN', 'Transformer'], required=True, 
                        help="評価に使用するモデルの構造 ('cnn' または 'transformer')")
    parser.add_argument('--weights', type=str, required=True, help="保存されたモデルの重みファイル (.pth) のパス")
    parser.add_argument('--ca', type=str, required=True, help="評価用 CA15-3 のデータフォルダパス")
    parser.add_argument('--cea', type=str, required=True, help="評価用 CEA のデータフォルダパス")

    # ==========================================
    # コマンドの全角スペース対策 ＆ ハイフン省略の処理
    # ==========================================
    raw_args_str = " ".join(sys.argv[1:])
    split_args = re.split(r'[　 \t]+', raw_args_str)

    valid_keys =['model', 'weights', 'ca', 'cea']
    processed_args =[]
    for arg in split_args:
        if not arg: continue
        if arg in valid_keys:
            processed_args.append(f'--{arg}')
        else:
            processed_args.append(arg)

    args = parser.parse_args(processed_args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using Device: {device}")

    # ==========================================
    # モデルのロード
    # ==========================================
    model_type = args.model.lower()
    if model_type == "cnn":
        model = WaveformCNN()
        trainer = CNN_Trainer(model, device)
    elif model_type == "transformer":
        model = WaveformTransformer()
        trainer = Transformer_Trainer(model, device)

    print(f"Loading weights from {args.weights}...")
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    model.eval()

    # ==========================================
    # 評価用データセットの準備
    # ==========================================
    print("Loading evaluation dataset...")
    eval_dataset = create_dataset([
        (args.ca, 0),
        (args.cea, 1)
    ])
    
    print(f"Dataset size: {len(eval_dataset)} samples")

    # ==========================================
    # 評価の実行
    # ==========================================
    print("\n--- Start Evaluation ---")
    auc = trainer.evaluate(eval_dataset)
    print(f"Evaluation AUC: {auc:.4f}")

if __name__ == "__main__":
    main()