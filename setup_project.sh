#!/usr/bin/env bash

# カレントディレクトリがリポジトリのルートであることを仮定

# ディレクトリ作成
mkdir -p data/images
mkdir -p data/masks
mkdir -p models
mkdir -p scripts

# スクリプトファイルをscriptsディレクトリに移動
# *.pyファイルが無い場合にエラーが出ないように、失敗時無視するため 2>/dev/null を付与
mv *.py scripts/ 2>/dev/null || true

# 必要に応じてREADMEやサンプル画像もscriptsに移動せず、そのままルートに残す場合は、このステップは不要
# mv README.md scripts/ 2>/dev/null || true
# mv sample_images scripts/ 2>/dev/null || true

echo "Project structure setup complete."
