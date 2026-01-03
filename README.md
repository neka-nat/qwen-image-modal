# qwen-image-modal

[Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) を [Modal](https://modal.com) の GPU 上で動かし、
ローカルのコマンドからプロンプトを渡して画像を生成するためのツールです。

- ローカルの依存は `modal` だけです。torch / diffusers などは Modal 側のイメージにだけ入ります。
- モデル重み（約 30 GB）は Modal の Volume にキャッシュし、初回だけダウンロードします。
- 推論は H100 上で bf16 で行い、生成した PNG をローカルに保存します。

> **ライセンスに注意**: Qwen-Image-2.1 は [Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE)
> で配布されており、**非商用利用に限定**されています。商用利用には別途 Qwen 側への申請が必要です。

## セットアップ

```bash
uv sync
uv run modal setup            # 初回のみ。Modal にログインしていない場合
```

## 使い方

```bash
# 1. モデル重みを Volume に取得（初回のみ。CPU コンテナで実行）
uv run modal run -m qwen_image_modal.app::download_model

# 2. 推論クラスをデプロイ（コードを変更したら再実行）
uv run modal deploy -m qwen_image_modal.app

# 3. 生成
uv run qwen-image-modal "A capybara wearing a wizard hat, oil painting"
uv run qwen-image-modal "..." --aspect 16:9 --seed 42
uv run qwen-image-modal "..." --size 1k -n 4          # 1K 解像度で 4 枚（seed, seed+1, ...）
uv run qwen-image-modal "a red maple leaf" --rgba      # 透過 PNG
uv run qwen-image-modal "Move it to a snowy mountain" --image outputs/xxx.png   # 編集
```

生成物は `./outputs/` に `PNG` と、同名の `JSON`（プロンプト・seed・ステップ数など）として保存されます。
同じ内容は PNG の `parameters` テキストチャンクにも埋め込まれます。

### オプション

| オプション | 既定値 | 説明 |
|---|---|---|
| `--aspect` | `1:1` | `1:1, 4:3, 3:4, 3:2, 2:3, 16:9, 9:16` |
| `--size` | `2k` | `2k` は公式の解像度表、`1k` は同じ比率で面積 1024² 相当 |
| `--width/--height` | なし | 32 の倍数で明示指定。`--aspect/--size` より優先 |
| `--steps` | `40` | Qwen 推奨値 |
| `--seed` | ランダム | 複数枚は `seed, seed+1, ...` |
| `-n/--num-images` | `1` | 生成枚数（1 枚ずつ順に生成） |
| `--cfg` | `1.0` | `true_cfg_scale`。1.0 でガイダンスなし（Qwen 推奨） |
| `--negative-prompt` | なし | `--cfg > 1` のときのみ有効 |
| `--rgba` | off | 透過画像用のプロンプトで包み、アルファ付き PNG で保存 |
| `--image PATH` | なし | 編集用の条件画像。複数指定可 |
| `-o/--out` | `outputs` | 出力ディレクトリ |

### 開発時

デプロイせずに、ログを流しながら試す場合:

```bash
uv run modal run -m qwen_image_modal.app --prompt "..." --aspect 16:9 --size 1k
```

## 環境変数

| 変数 | 既定値 | 説明 |
|---|---|---|
| `QWEN_IMAGE_MODAL_APP` | `qwen-image-modal` | Modal のアプリ名 |
| `QWEN_IMAGE_MODAL_GPU` | `H100` | デプロイ時の GPU 種別（bf16 で 32 GB 超必要） |

## リモート側の Python について

Modal イメージは Python 3.12 でビルドしています（ローカルは 3.13 のままで問題ありません）。
ワークスペース既定の Image Builder（2023.12）が Python 3.13 に対応していないためです。
3.13 に揃えたい場合は、Modal ダッシュボードの Workspace Settings → Image Config で
Image Builder Version を `2024.10` 以降に上げてから、`app.py` の `python_version` を `"3.13"` に変更してください。

## バージョン固定

`src/qwen_image_modal/config.py` で次を固定しています。

- モデル: `Qwen/Qwen-Image-2.1` の HF コミット
- diffusers: `QwenImage21Pipeline` はリリース版 (0.40.0) に未収録のため、main のコミット SHA
- Modal イメージ内の torch / transformers / accelerate / huggingface-hub / pillow のバージョンは `app.py`

## 構成

```
src/qwen_image_modal/
├── config.py   # モデル・バージョン固定、解像度プリセット
├── app.py      # Modal App / Image / Volume / Inference クラス / modal run 用エントリポイント
└── cli.py      # ローカル CLI（デプロイ済みクラスを modal.Cls.from_name で呼ぶ）
```
