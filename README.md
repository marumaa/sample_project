# 民泊事業向け社内Chatbot

## 概要
このアプリケーションは、民泊物件の詳細情報への社内問い合わせ対応を自動化するためのChatbotです。PDFファイルから情報を抽出し、RAG（Retrieval-Augmented Generation）方式で質問に回答します。

## 機能
- PDFファイルのアップロードと処理
  - ローカルからのアップロード
  - Googleドライブからのアップロード
- ベクトルデータベースによる情報検索
- 自然言語による質問応答
- ユーザー認証機能

## セットアップ
1. 必要なパッケージのインストール:
```bash
pip install -r requirements.txt
```

2. 環境変数の設定:
`.env`ファイルにOpenAI APIキーを設定してください。
```
OPENAI_API_KEY=your_openai_api_key_here
```

3. 認証設定:
`config.yaml`ファイルの認証情報を設定してください。

4. Googleドライブ連携の設定:
   - [Google Cloud Console](https://console.cloud.google.com/) で新しいプロジェクトを作成してください。
   - Google Drive APIを有効にしてください。
   - OAuth 2.0クライアントIDを作成し、認証情報をダウンロードしてください。
   - ダウンロードしたJSONファイルを`credentials.json`として保存してください。

## 使用方法
1. アプリケーションの起動:
```bash
streamlit run app.py
```

2. ログイン:
- 設定したユーザー名とパスワードでログイン

3. PDFファイルのアップロード:
   - ローカルからのアップロード:
     - サイドバーの「ローカルからアップロード」からPDFファイルを選択
   - Googleドライブからのアップロード:
     - 「Googleドライブに接続」ボタンをクリック
     - 初回利用時はGoogleアカウントでの認証が必要
     - ファイルを選択し、「選択したファイルを処理」ボタンをクリック

4. 質問:
- メイン画面で質問を入力

## 注意事項
- 初回起動時は、PDFファイルをアップロードしてから質問を行ってください
- 機密情報は適切に管理してください
- Googleドライブ連携には、Google Cloudプロジェクトの設定が必要です

# フロント対応チャットボット

## 環境構築

1. 仮想環境の作成と有効化
```bash
python -m venv venv
# Windowsの場合
.\venv\Scripts\Activate.ps1
# macOSとLinuxの場合
source venv/bin/activate
```

2. 必要なパッケージのインストール
```bash
pip install -r requirements.txt
```

3. 環境変数の設定
`.env`ファイルをプロジェクトのルートディレクトリに作成し、以下の内容を追加してください：
```
# OpenAI APIキー
OPENAI_API_KEY=ここにあなたのOpenAI APIキーを入力してください
```

## アプリケーションの実行

```bash
streamlit run app.py
```

## デプロイ方法

このアプリケーションは以下のプラットフォームでデプロイすることができます：

### Streamlit Cloud（推奨）
1. GitHubにプロジェクトをアップロード
2. https://share.streamlit.io/ でアカウント作成
3. GitHubリポジトリを接続

### Heroku
プロジェクトには既にHerokuデプロイ用の`Procfile`と`runtime.txt`が含まれています。

### Render
プロジェクトには既にRenderデプロイ用の`render.yaml`が含まれています。

## 注意事項

- APIキーは`.env`ファイルで管理し、Gitリポジトリにはコミットしないようにしてください。
- 本番環境では、プラットフォームの環境変数設定機能を使用してAPIキーを設定してください。 