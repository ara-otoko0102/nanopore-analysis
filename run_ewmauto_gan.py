from EWMAuto_kawai import EWMAuto
import sys

# =========================
# コマンドライン引数
# =========================
# 使い方:
# python run_ewmauto.py data.bin [control.bin] [output_folder] [output_prefix]
#
# 引数:
#   data.bin:      解析対象のデータファイル
#   control.bin:   (オプション) コントロールデータファイル。指定しない場合は "None" と入力
#   output_folder: (オプション) 出力先フォルダ名 (デフォルト: EWMAuto_output)
#   output_prefix: (オプション) 出力ファイル名の接頭辞 (デフォルト: EWMAuto)

if len(sys.argv) < 2:
    print("使い方:")
    print("python run_ewmauto.py data.bin [control.bin] [output_folder] [output_prefix]")
    print('コントロールなしの場合: python run_ewmauto.py data.bin None')
    sys.exit()

data_path = sys.argv[1]

# control.bin のパス
# "None" という文字列が渡された場合もコントロールなしと判断
if len(sys.argv) >= 3 and sys.argv[2].lower() != 'none':
    control_path = sys.argv[2]
else:
    control_path = None

# 出力フォルダ名
if len(sys.argv) >= 4:
    output_folder = sys.argv[3]
else:
    output_folder = "EWMAuto_output"

# 出力ファイル名の接頭辞
if len(sys.argv) >= 5:
    output_prefix = sys.argv[4]
else:
    output_prefix = "EWMAuto"


# =========================
# EWMAuto インスタンス作成
# =========================
ewmauto = EWMAuto(
    path_control=control_path,      # コントロールなしなら None
    data_scale_control=1.0,
    whether_blockage_control=True,
    sampling_rate=250000,            # 250 kHz
    section_length=30,               # ms
    max_threshold = 2000,
    detection_thr=7,
    log=True
)

# =========================
# 実行
# =========================
ewmauto.run(
    path_data=data_path,
    data_scale=1e13,
    whether_blockage_data=True,
    output_folder_name=output_folder,  # 引数で指定したフォルダ名を使用
    output_file_prefix=output_prefix,  # 引数で指定した接頭辞を使用
    EWMA_weight=0.9999,
    save_trace=True,
    max_events=10000
)

