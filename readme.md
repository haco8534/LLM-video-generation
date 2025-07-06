# 必要アプリケーション（事前にインストール）
- FFmpeg（動画編集）
- VoiceVox（音声合成）

## 必要なAPIキー
- OpenAI API key
- Pixabay API key

ルートディレクトリに .env ファイルを作成し、以下のように記述してください：

OPENAI_API_KEY=your_openai_api_key
PIXABAY_API_KEY=your_pixabay_api_key

# セットアップ手順（Windows）

## 1. 仮想環境の作成
python -m venv venv

## 2. 仮想環境のアクティベート
venv\Scripts\activate

## 3. ライブラリのインストール
pip install -r requirements.txt