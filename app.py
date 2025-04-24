import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import os
import io
import tempfile
import json
import datetime
import shutil
from dotenv import load_dotenv
from pathlib import Path
import fitz  # PyMuPDF
import faiss
import numpy as np
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage, Document
from langchain.callbacks.base import BaseCallbackHandler
from typing import Any, Dict, List, Union
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import time
import heapq

# Googleドライブ連携用
import pickle
import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

# 環境変数の読み込み
load_dotenv()

# OpenAI APIキーの確認 - 環境変数から読み込む
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("OpenAI APIキーが設定されていません。")
    st.stop()
else:
    # 読み込んだAPIキーの一部を表示（デバッグ用、本番では削除）
    masked_key = OPENAI_API_KEY[:5] + "..." + OPENAI_API_KEY[-4:]
    st.info(f"読み込まれたAPIキー (一部): {masked_key}")

# 認証設定の読み込み
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

# Streamlitストリーミング用のコールバックハンドラ
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text
        self.container.markdown(self.text)

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text)
        
    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        pass

# セッション状態の初期化
if 'messages' not in st.session_state:
    st.session_state.messages = []
    
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

if 'use_faster_model' not in st.session_state:
    st.session_state.use_faster_model = False

if 'top_k_documents' not in st.session_state:
    st.session_state.top_k_documents = 4

if 'use_advanced_search' not in st.session_state:
    st.session_state.use_advanced_search = False

if 'use_document_reranking' not in st.session_state:
    st.session_state.use_document_reranking = False
    
if 'use_document_compression' not in st.session_state:
    st.session_state.use_document_compression = False

# 処理済みファイルの記録
PROCESSED_FILES_JSON = "processed_files.json"
VECTOR_STORE_DIR = "vector_store"

def load_processed_files():
    if os.path.exists(PROCESSED_FILES_JSON):
        with open(PROCESSED_FILES_JSON, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []

def save_processed_file(file_id, file_name, file_type="pdf", source="local"):
    processed_files = load_processed_files()
    
    # リストまたは古い形式の場合は新しい形式に変換
    if isinstance(processed_files, list) and all(isinstance(item, str) for item in processed_files):
        # 古い形式（ID文字列のリスト）から新しい形式（辞書のリスト）に変換
        processed_files = [{"id": item, "name": "不明", "type": "pdf", "source": "unknown", "date": "不明"} for item in processed_files]
    
    # すでに存在するファイルIDかチェック
    file_exists = False
    for file in processed_files:
        if isinstance(file, dict) and file.get("id") == file_id:
            file_exists = True
            # 既存のファイル情報を更新
            file["name"] = file_name
            file["type"] = file_type
            file["source"] = source
            file["date"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            break
    
    # 存在しない場合は新規追加
    if not file_exists:
        processed_files.append({
            "id": file_id,
            "name": file_name,
            "type": file_type,
            "source": source,
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    # ファイルに保存
    with open(PROCESSED_FILES_JSON, 'w') as f:
        json.dump(processed_files, f, indent=2)
    
    return processed_files

# 処理済みファイルの削除
def delete_processed_file(file_id):
    processed_files = load_processed_files()
    
    # 新しい形式に変換（互換性のため）
    if isinstance(processed_files, list) and len(processed_files) > 0:
        if all(isinstance(item, str) for item in processed_files):
            processed_files = [{"id": item, "name": "不明", "type": "pdf", "source": "unknown", "date": "不明"} for item in processed_files]
    
    # 指定されたIDのファイルを削除
    processed_files = [file for file in processed_files if file.get("id") != file_id]
    
    # ファイルに保存
    with open(PROCESSED_FILES_JSON, 'w') as f:
        json.dump(processed_files, f, indent=2)
    
    return processed_files

# すべての処理済みファイルを削除
def delete_all_processed_files():
    # 処理済みファイル記録を空にする
    with open(PROCESSED_FILES_JSON, 'w') as f:
        json.dump([], f)
    
    # ベクトルストアも削除
    if os.path.exists(VECTOR_STORE_DIR):
        shutil.rmtree(VECTOR_STORE_DIR)
        
    # 会話履歴をリセット
    if 'qa_memory' in st.session_state:
        del st.session_state['qa_memory']
    
    # システムメッセージを追加
    st.session_state.messages.append({"role": "system", "content": "すべてのファイルを削除しました。チャット履歴もリセットされました。"})
    st.session_state.conversation_history = []
    
    return True

# ベクトルストアの再構築（削除したファイルの情報を除外）
def rebuild_vector_store(file_id_to_exclude=None):
    # ベクトルストアを削除
    if os.path.exists(VECTOR_STORE_DIR):
        shutil.rmtree(VECTOR_STORE_DIR)
    
    # 会話履歴をリセット
    if 'qa_memory' in st.session_state:
        del st.session_state['qa_memory']
    st.session_state.conversation_history = []
    
    # 残りのファイルを再処理
    processed_files = load_processed_files()
    if not processed_files:
        return True
    
    # 選択的削除の場合は、指定されたファイルを除外
    if file_id_to_exclude:
        processed_files = [file for file in processed_files if file.get("id") != file_id_to_exclude]
    
    # ここで再処理のロジックを実装
    # ただし、元のファイルデータが保存されていないため、実際の再処理は難しい
    # この例では、ベクトルストアのリセットのみを行い、ユーザーに再アップロードを促す
    
    return True

# 処理済みファイルの一覧表示
def display_processed_files():
    processed_files = load_processed_files()
    
    # 新しい形式に変換（互換性のため）
    if isinstance(processed_files, list) and len(processed_files) > 0:
        if all(isinstance(item, str) for item in processed_files):
            processed_files = [{"id": item, "name": "不明", "type": "pdf", "source": "unknown", "date": "不明"} for item in processed_files]
    
    if not processed_files:
        st.info("処理済みのファイルはありません")
        return
    
    # 日付で降順ソート（最新のものが上に来るように）
    try:
        processed_files = sorted(processed_files, key=lambda x: x.get("date", ""), reverse=True)
    except:
        # ソートに失敗した場合は元の順序のまま
        pass
    
    # 全削除ボタン
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader(f"処理済みファイル ({len(processed_files)}件)")
    with col2:
        if st.button("全ファイル削除", type="primary", use_container_width=True):
            if delete_all_processed_files():
                st.success("すべてのファイルを削除しました")
                st.rerun()  # 画面を再読み込み
    
    # ファイル一覧をテーブルで表示
    for i, file in enumerate(processed_files):
        if isinstance(file, dict):
            name = file.get("name", "不明")
            date = file.get("date", "不明")
            source = file.get("source", "不明")
            file_id = file.get("id", "")
            
            # 表示用にソースを日本語に変換
            if source == "google_drive":
                source_display = "Googleドライブ"
            elif source == "local":
                source_display = "ローカル"
            else:
                source_display = source
            
            # ファイル情報とボタンを横に並べる
            col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
            with col1:
                st.write(f"**{name}**")
            with col2:
                st.write(source_display)
            with col3:
                st.write(date)
            with col4:
                if st.button("削除", key=f"delete_{i}", use_container_width=True):
                    delete_processed_file(file_id)
                    rebuild_vector_store(file_id)
                    st.success(f"ファイル「{name}」を削除しました")
                    st.session_state.messages.append({"role": "system", "content": f"ファイル「{name}」を削除しました。チャット履歴はリセットされました。"})
                    st.rerun()  # 画面を再読み込み
            
            # 区切り線
            st.markdown("---")

# ベクトルDBの初期化
def init_vector_db():
    embeddings = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY,
        model="text-embedding-3-small"  # 最新の高性能エンベディングモデルを使用
    )
    
    if os.path.exists("vector_store"):
        try:
            # FAISS.load_localを使用してロード
            base_vector_store = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)
            
            # ベクトルストアの基本情報を表示
            index_size = len(base_vector_store.index_to_docstore_id) if hasattr(base_vector_store, 'index_to_docstore_id') else "不明"
            st.sidebar.success(f"ベクトルストアを読み込みました（サイズ: {index_size} チャンク）")
            
            # 高度な検索を有効にした場合、追加設定
            if st.session_state.use_advanced_search:
                # 必要に応じてインデックスを再構築（通常は不要）
                # ここではロード済みのベクトルストアを使用
                pass
                
            return base_vector_store
        except Exception as e:
            st.error(f"ベクトルストアの読み込み中にエラーが発生しました: {e}")
            import traceback
            st.error(traceback.format_exc())
            return None
    return None

# ファイルの種類を判断
def get_file_type(file):
    if hasattr(file, 'name'):
        file_ext = Path(file.name).suffix.lower()
        if file_ext == '.pdf':
            return 'pdf'
        elif file_ext == '.md':
            return 'markdown'
        elif file_ext == '.json':
            return 'json'
    return None

# PDFファイルの処理
def process_pdf(file):
    try:
        doc = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        st.error(f"PDFファイルの処理中にエラーが発生しました: {e}")
        return None

# マークダウンファイルの処理
def process_markdown(file):
    try:
        # マークダウンファイルはテキストファイルとして読み込むだけでOK
        content = file.read().decode('utf-8')
        return content
    except Exception as e:
        st.error(f"マークダウンファイルの処理中にエラーが発生しました: {e}")
        return None

# JSONファイルの処理
def process_json(file):
    try:
        # JSONファイルを読み込み
        content = file.read().decode('utf-8')
        # JSONを辞書や配列にパース
        import json
        json_data = json.loads(content)
        # JSONデータをテキスト形式に変換
        if isinstance(json_data, dict) or isinstance(json_data, list):
            # 整形して表示
            text = json.dumps(json_data, ensure_ascii=False, indent=2)
            # デバッグ情報を表示
            with st.sidebar.expander("JSON構造", expanded=False):
                if isinstance(json_data, dict):
                    st.write(f"キー数: {len(json_data.keys())}")
                    st.write("トップレベルキー:", list(json_data.keys())[:10])
                elif isinstance(json_data, list):
                    st.write(f"項目数: {len(json_data)}")
                    if len(json_data) > 0 and isinstance(json_data[0], dict):
                        st.write("最初の項目のキー:", list(json_data[0].keys())[:10])
            return text
        else:
            st.error("JSONデータの形式が不正です")
            return None
    except json.JSONDecodeError:
        st.error("JSONデータの解析に失敗しました。有効なJSON形式であることを確認してください。")
        return None
    except Exception as e:
        st.error(f"JSONファイルの処理中にエラーが発生しました: {e}")
        return None

# ファイル処理（ファイルタイプに応じて適切な処理を行う）
def process_file(file):
    file_type = get_file_type(file)
    
    # ファイル処理の開始を表示
    st.sidebar.info(f"ファイル処理開始: {file.name} ({file_type or '不明'}タイプ)")
    
    if file_type == 'pdf':
        text = process_pdf(file)
        file_type_str = "pdf"
    elif file_type == 'markdown':
        text = process_markdown(file)
        file_type_str = "markdown"
    elif file_type == 'json':
        text = process_json(file)
        file_type_str = "json"
    else:
        st.error("サポートされていないファイル形式です。PDF、マークダウン(.md)、またはJSONファイルをアップロードしてください。")
        return None
    
    # テキスト抽出の結果を表示
    if text:
        st.sidebar.success(f"テキスト抽出成功: {len(text)} 文字")
        # テキストサンプルを表示（デバッグ用）
        with st.sidebar.expander("抽出テキストサンプル", expanded=False):
            st.write(text[:500] + "..." if len(text) > 500 else text)
    else:
        st.sidebar.error("テキスト抽出に失敗しました")
        
    return text, file_type_str

# テキスト処理とベクトル化
def process_text(text, vector_store, file_name=None, file_type=None):
    if not text:
        return vector_store # テキストがなければ何もしない
    try:
        # ベクターストアの検証
        if vector_store is not None:
            st.sidebar.success(f"既存のベクトルストアを使用します（サイズ: {len(vector_store.index_to_docstore_id) if hasattr(vector_store, 'index_to_docstore_id') else '不明'}）")
        else:
            st.sidebar.info("新規ベクトルストアを作成します")
        
        # チャンクサイズとオーバーラップの設定
        # 日本語テキストに適した値に調整
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,  # より小さいチャンクサイズに調整（検索精度向上のため）
            chunk_overlap=150,  # オーバーラップを増やして文脈の連続性を確保
            separators=["\n\n", "\n", "。", "、", " ", ""],  # 日本語に適した区切り文字
            length_function=len  # 日本語の文字数をカウント
        )
        
        # テキストをチャンクに分割
        texts = text_splitter.split_text(text)
        
        # チャンク化の結果を表示
        st.sidebar.success(f"テキストを {len(texts)} チャンクに分割しました")
        
        # チャンク内容の確認（デバッグ用）
        with st.sidebar.expander("チャンク内容サンプル（最初の3件）", expanded=False):
            for i, chunk in enumerate(texts[:3]):
                st.write(f"チャンク {i+1}: {chunk[:100]}...")
            if len(texts) > 3:
                st.write(f"...他 {len(texts) - 3} チャンク")
        
        # エンベディング設定
        embeddings = OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY,
            model="text-embedding-3-small"  # 最新の高性能エンベディングモデルを使用
        )
        
        # メタデータに情報追加
        metadatas = []
        for i, chunk in enumerate(texts):
            metadata = {
                "source": file_name or f"chunk_{i}",
                "chunk_id": i,
                "file_type": file_type or "text",
                "chunk_size": len(chunk),
                "chunk_text": chunk[:100]  # チャンクのプレビューをメタデータに追加
            }
            metadatas.append(metadata)
        
        if not texts:
             st.warning("テキストの分割結果が空です。処理をスキップします。")
             return vector_store

        # ベクトルストアに追加
        if vector_store is None:
            vector_store = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
            st.sidebar.success("新しいベクトルストアを作成しました")
        else:
            vector_store.add_texts(texts, metadatas=metadatas)
            st.sidebar.success("既存のベクトルストアにテキストを追加しました")
        
        # ベクトルストアを保存
        vector_store.save_local("vector_store")
        
        # ベクトルストアの統計情報
        with st.sidebar.expander("ベクトルストア情報", expanded=False):
            index_size = len(vector_store.index_to_docstore_id) if hasattr(vector_store, 'index_to_docstore_id') else "不明"
            st.write(f"ベクトルストアのサイズ: {index_size} チャンク")
            st.write(f"最後に追加したドキュメント: {file_name or '不明'}")
            st.write(f"最後に追加したチャンク数: {len(texts)}")
        
        return vector_store
    except Exception as e:
        st.error(f"テキストの処理とベクトル化中にエラーが発生しました: {e}")
        import traceback
        st.error(traceback.format_exc())
        return vector_store

# Googleドライブ認証とファイル一覧取得
def authenticate_google_drive():
    SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
    creds = None
    
    # オフラインチェック - 接続テスト
    if not check_internet_connection():
        st.error("インターネット接続がオフラインです。ネットワーク接続を確認してください。")
        return None
    
    # トークンがある場合はロード
    if os.path.exists('token.pickle'):
        try:
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)
        except Exception as e:
            st.error(f"認証トークンの読み込みに失敗しました: {e}")
            # トークンファイルが破損している可能性があるため削除
            try:
                os.remove('token.pickle')
                st.warning("認証トークンをリセットしました。再度認証を行ってください。")
            except:
                pass
    
    # 有効なトークンがない場合は認証
    if not creds or not creds.valid:
        try:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except Exception as e:
                    st.error(f"トークンの更新に失敗しました: {e}")
                    # トークンの更新に失敗した場合、トークンを削除して再認証
                    if os.path.exists('token.pickle'):
                        os.remove('token.pickle')
                    creds = None
            
            if not creds:
                if not os.path.exists('credentials.json'):
                    st.error("credentials.jsonファイルが見つかりません。Googleドライブ連携には必須です。")
                    return None
                    
                # ローカル開発用：ブラウザは開かないが、ローカルサーバーを使用
                st.info("Googleアカウントでの認証が必要です。ターミナルに表示されるURLをブラウザで開き、表示された認証コードを入力してください。")
                try:
                    from google_auth_oauthlib.flow import InstalledAppFlow
                    flow = InstalledAppFlow.from_client_secrets_file(
                        'credentials.json', SCOPES)
                        
                    # ローカルサーバーを使用するが、ブラウザは自動で開かない
                    creds = flow.run_local_server(port=0, open_browser=False)
                except Exception as e:
                    st.error(f"認証プロセス中にエラーが発生しました: {e}")
                    return None
                
            # トークンを保存
            try:
                with open('token.pickle', 'wb') as token:
                    pickle.dump(creds, token)
            except Exception as e:
                st.warning(f"トークンの保存に失敗しました: {e}")
                # トークンの保存に失敗しても続行可能
        except Exception as auth_error:
            st.error(f"認証処理中に予期せぬエラーが発生しました: {auth_error}")
            return None
    
    # APIサービスの構築
    try:
        service = build('drive', 'v3', credentials=creds)
        # 接続テスト - about.getでドライブ情報を取得
        service.about().get(fields="user").execute()
        return service
    except HttpError as error:
        st.error(f"Google API接続エラー: {error}")
        return None
    except Exception as e:
        st.error(f"Google APIの初期化に失敗しました: {e}")
        return None

# インターネット接続チェック
def check_internet_connection():
    import socket
    try:
        # Googleのパブリックリゾルバーに接続テスト（DNS）
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        try:
            # バックアップとしてGoogle.comにHTTP接続テスト
            socket.create_connection(("www.google.com", 80), timeout=3)
            return True
        except OSError:
            return False

# ファイルリストを取得（リトライロジック付き）
def get_drive_files(service, folder_id=None, file_types=None, max_retries=3):
    if not service:
        st.error("Google Driveサービスが初期化されていません。")
        return []
        
    retry_count = 0
    last_error = None
    
    while retry_count < max_retries:
        try:
            query = "'me' in owners"
            if folder_id:
                query += f" and '{folder_id}' in parents"
            
            # ファイルタイプとファイル名の両方で検索（MIMEタイプまたはファイル名の拡張子）
            if file_types:
                # MIMEタイプによる検索条件
                mime_query = " or ".join([f"mimeType='{file_type}'" for file_type in file_types])
                
                # ファイル名による検索条件（.md、.mdxなどで終わるファイル）
                # Google Drive APIはファイル名に対して正規表現をサポートしていないが、
                # "contains"を使用して簡易的な検索を行う
                name_query = " or ".join([f"name contains '.md'" for _ in range(1)])
                
                # MIMEタイプまたはファイル名で検索
                query += f" and ({mime_query} or {name_query})"
            
            # デバッグ用：最終的なクエリを表示
            st.sidebar.info(f"検索クエリ: {query}")
            
            results = service.files().list(
                q=query,
                pageSize=100,  # 取得数を増やす
                fields="nextPageToken, files(id, name, mimeType, webViewLink)"
            ).execute()
            
            files = results.get('files', [])
            
            # ページングがある場合の処理（多数のファイルがある場合）
            next_page_token = results.get('nextPageToken')
            while next_page_token:
                results = service.files().list(
                    q=query,
                    pageSize=100,
                    pageToken=next_page_token,
                    fields="nextPageToken, files(id, name, mimeType, webViewLink)"
                ).execute()
                files.extend(results.get('files', []))
                next_page_token = results.get('nextPageToken')
                
            # デバッグ用：検索結果の詳細を表示
            if files:
                st.sidebar.expander("検索結果の詳細", expanded=False).json(
                    [{"name": f["name"], "mimeType": f["mimeType"]} for f in files]
                )
            else:
                st.sidebar.warning("検索結果: ファイルが見つかりませんでした")
                
            return files
            
        except HttpError as error:
            last_error = error
            error_str = str(error)
            # 一時的なエラーの場合のみリトライ
            if "503" in error_str or "500" in error_str or "timeout" in error_str.lower() or "connection" in error_str.lower():
                retry_count += 1
                st.warning(f"Google Driveへの接続エラーが発生しました。再試行します({retry_count}/{max_retries})...")
                time.sleep(2)  # リトライ前の待機時間
            else:
                # 認証エラーや権限エラーなど、リトライしても解決しないエラー
                break
        except Exception as e:
            last_error = e
            retry_count += 1
            st.warning(f"予期せぬエラーが発生しました。再試行します({retry_count}/{max_retries})...")
            time.sleep(2)
    
    # すべてのリトライが失敗した場合
    st.error(f'Googleドライブの読み込み中にエラーが発生しました: {last_error}')
    
    # エラーの種類に応じたヘルプメッセージを表示
    error_str = str(last_error).lower()
    if "unable to find the server" in error_str:
        st.error("サーバーが見つかりません。インターネット接続を確認してください。")
    elif "ssl" in error_str or "certificate" in error_str or "decrypt" in error_str:
        st.error("SSL/TLS接続に問題があります。セキュリティソフトやプロキシの設定を確認してください。また、システム日時が正確であることも確認してください。")
    elif "quota" in error_str:
        st.error("Google APIのクォータを超えました。しばらく待ってから再試行してください。")
    elif "permission" in error_str or "access" in error_str:
        st.error("アクセス権限がありません。Googleアカウントの権限設定を確認してください。")
        
    return []

# ファイルをダウンロード（リトライロジック付き）
def download_file(service, file_id, max_retries=3):
    if not service:
        st.error("Google Driveサービスが初期化されていません。")
        return None
        
    retry_count = 0
    last_error = None
    
    while retry_count < max_retries:
        try:
            request = service.files().get_media(fileId=file_id)
            file_content = io.BytesIO()
            downloader = MediaIoBaseDownload(file_content, request)
            done = False
            
            # ダウンロード進捗を表示
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            while done is False:
                status, done = downloader.next_chunk()
                if status:
                    progress = int(status.progress() * 100)
                    progress_bar.progress(progress / 100)
                    status_text.text(f"ダウンロード中: {progress}%")
            
            # 進捗表示をクリア
            status_text.empty()
            progress_bar.empty()
            
            file_content.seek(0)
            return file_content
            
        except HttpError as error:
            last_error = error
            error_str = str(error)
            # 一時的なエラーの場合のみリトライ
            if "503" in error_str or "500" in error_str or "timeout" in error_str.lower() or "connection" in error_str.lower():
                retry_count += 1
                st.warning(f"ファイルのダウンロード中にエラーが発生しました。再試行します({retry_count}/{max_retries})...")
                time.sleep(2)  # リトライ前の待機時間
            else:
                # 認証エラーや権限エラーなど、リトライしても解決しないエラー
                break
        except Exception as e:
            last_error = e
            retry_count += 1
            st.warning(f"ファイルのダウンロード中に予期せぬエラーが発生しました。再試行します({retry_count}/{max_retries})...")
            time.sleep(2)
    
    # すべてのリトライが失敗した場合
    st.error(f'ファイルのダウンロード中にエラーが発生しました: {last_error}')
    return None

# フォルダの一覧を取得
def get_drive_folders(service):
    try:
        query = "'me' in owners and mimeType='application/vnd.google-apps.folder'"
        results = service.files().list(
            q=query,
            pageSize=50,
            fields="nextPageToken, files(id, name)"
        ).execute()
        
        folders = results.get('files', [])
        return folders
    except HttpError as error:
        st.error(f'Googleドライブのフォルダ一覧取得中にエラーが発生しました: {error}')
        return []

# すべてのPDFファイルを一括処理
def batch_process_drive_files(service, folder_id=None):
    processed_files_data = load_processed_files()
    
    # 処理済みファイルIDのリストを抽出（旧形式と新形式の両方に対応）
    processed_ids = []
    for item in processed_files_data:
        if isinstance(item, str):
            processed_ids.append(item)
        elif isinstance(item, dict) and "id" in item:
            processed_ids.append(item["id"])
    
    # PDFとマークダウンの両方を対象にする
    files = get_drive_files(
        service, 
        folder_id, 
        file_types=[
            'application/pdf',        # PDF
            'text/markdown',          # Markdown
            'text/x-markdown',        # 別のMD MIMEタイプ
            'text/plain',             # プレーンテキスト（.mdファイルが時々このタイプに）
            'application/octet-stream' # バイナリデータ（一部のドライブではこれに）
        ]
    )
    
    # マークダウンファイルの拡張子でフィルタリング
    # PDFはMIMEタイプが正確なので追加フィルタリングは不要
    markdown_extensions = ['.md', '.markdown', '.mdown', '.mkdn', '.mdx']
    filtered_files = []
    
    for file in files:
        # PDFファイルはそのまま追加
        if file['mimeType'] == 'application/pdf':
            filtered_files.append(file)
        # ファイル名で.mdなどのマークダウンファイルをチェック
        elif any(file['name'].lower().endswith(ext) for ext in markdown_extensions):
            filtered_files.append(file)
        # "text/markdown" または "text/x-markdown" MIMEタイプならそのまま追加
        elif file['mimeType'] in ['text/markdown', 'text/x-markdown']:
            filtered_files.append(file)
    
    # デバッグ情報表示
    st.sidebar.success(f"検出されたファイル数: {len(filtered_files)}")
    
    if not filtered_files:
        st.info("処理対象のPDFまたはマークダウンファイルが見つかりませんでした。")
        return 0
    
    # 処理済みのファイルをスキップ
    files_to_process = [f for f in filtered_files if f['id'] not in processed_ids]
    
    if not files_to_process:
        st.info("すべてのファイルはすでに処理済みです。")
        return 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    vector_store = init_vector_db()
    processed_count = 0
    
    for i, file in enumerate(files_to_process):
        status_text.text(f"処理中: {file['name']} ({i+1}/{len(files_to_process)})")
        
        try:
            file_content = download_file(service, file['id'])
            if not file_content:
                st.warning(f"ファイル「{file['name']}」のダウンロードに失敗しました。")
                continue
                
            # ファイルの種類を判断
            file_type = "pdf"  # デフォルト
            if file['mimeType'] in ['text/markdown', 'text/x-markdown'] or any(file['name'].lower().endswith(ext) for ext in markdown_extensions):
                file_type = "markdown"
            
            # 一時ファイルとして保存して処理
            suffix = '.md' if file_type == "markdown" else '.pdf'
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(file_content.getvalue())
                temp_file_path = temp_file.name
            
            try:
                # ファイルを開いて処理
                with open(temp_file_path, 'rb') as f:
                    # ファイルタイプに応じた処理
                    if file_type == "markdown":
                        text = process_markdown(io.BytesIO(f.read()))
                        st.info(f"マークダウンファイルとして処理: {file['name']}")
                    else:  # PDF
                        text = process_pdf(io.BytesIO(f.read()))
                        st.info(f"PDFファイルとして処理: {file['name']}")
                
                # テキスト処理とベクトル化
                if text:
                    vector_store = process_text(text, vector_store, file_name=file['name'], file_type=file_type)
                    if vector_store:
                        # 処理済みファイル情報を保存
                        save_processed_file(file['id'], file['name'], file_type, "google_drive")
                        processed_count += 1
                        st.sidebar.success(f"処理完了: {file['name']} ({file_type})")
                else:
                    st.warning(f"ファイル「{file['name']}」からテキストを抽出できませんでした。")
            except Exception as e:
                st.error(f"ファイル処理中にエラーが発生しました: {e}")
            finally:
                # 一時ファイルを削除
                os.unlink(temp_file_path)
        except Exception as e:
            st.error(f"ファイル {file['name']} の処理中に予期せぬエラーが発生しました: {e}")
        
        # 進捗バーを更新
        progress_bar.progress((i + 1) / len(files_to_process))
    
    status_text.text(f"処理完了: {processed_count}個のファイルを処理しました。")
    return processed_count

# 曖昧な質問に対するフォローアップ質問を含むプロンプトテンプレート
CONDENSE_QUESTION_PROMPT = """
あなたは民泊施設の情報を提供する対話型アシスタントです。
利用者からの質問に正確に答えるために、曖昧な質問や情報が不足している質問については、丁寧に追加情報を求めてください。

以下は会話の履歴と現在の質問です：

会話履歴：
{chat_history}

現在の質問：{question}

質問が曖昧または不完全な場合、追加情報を求めるフォローアップ質問をしてください。
質問が具体的で明確な場合は、そのまま質問を返してください。

回答：
"""

# 改善されたQAプロンプトテンプレート
QA_PROMPT = """
あなたは民泊施設の情報を提供する専門的なアシスタントです。
提供された情報源に基づいて、正確かつ丁寧に回答してください。

以下のドキュメントを参照してください:
{context}

ユーザーの質問: {question}

回答の際の指示:
1. 情報源に記載がある内容のみを使って回答してください。
2. 情報源に記載されている事実は、明確かつ具体的に伝えてください。
3. 情報源に記載がない場合は、「申し訳ありませんが、その情報は資料に記載がありません」と明確に伝えてください。
4. 回答を作る際に、文章の重複や繰り返しを避けてください。
5. 各物件や部屋の情報は、IDや番号などで具体的に回答してください。
6. 必要に応じて、箇条書きや表形式で整理して回答してください。
7. 丁寧な言葉遣いを心がけてください。

特に重要: 情報源にない知識で回答を補足しないでください。必ず提供された情報源の内容に基づいて回答してください。

回答:
"""

# ドキュメントの再ランキング
def rerank_documents(documents, query, top_n=None):
    if not documents:
        return []
    
    if top_n is None:
        top_n = len(documents)
    
    # 簡易的な再ランキング（本番では専用のrerankerを使用するとより効果的）
    # ドキュメントに一意のIDを付与して、比較時の衝突を避ける
    scored_docs = []
    
    for i, doc in enumerate(documents):
        # スコアリングロジック（実際にはより洗練された方法を使用）
        # 現在は単純に元のスコアを使用
        score = getattr(doc, "score", 0.5)  # デフォルトスコア
        
        # スコア、一意のID、ドキュメントのタプルを保存
        # 同じスコアの場合は元の順序を維持するためにiを使用
        scored_docs.append((score, i, doc))
    
    # スコアでソート（降順）
    sorted_docs = sorted(scored_docs, key=lambda x: (-x[0], x[1]))
    
    # 上位N件のドキュメントを取得
    top_docs = [doc for _, _, doc in sorted_docs[:top_n]]
    
    return top_docs

# ドキュメントの重複削除と圧縮
def compress_documents(documents, query):
    if not documents:
        return []
    
    try:
        # ドキュメントのコンテンツベースでの重複削除
        seen_content = set()
        unique_docs = []
        
        for doc in documents:
            content = doc.page_content.strip() if hasattr(doc, 'page_content') else ""
            if len(content) > 50:  # 短すぎるコンテンツは比較しない
                content_hash = hash(content)
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    unique_docs.append(doc)
            else:
                unique_docs.append(doc)
        
        # LLMを使用してドキュメントを圧縮（クエリに関連する部分のみ抽出）
        if st.session_state.use_document_compression and len(unique_docs) > 2:
            try:
                llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0, model_name="gpt-3.5-turbo")
                compressor = LLMChainExtractor.from_llm(llm)
                compressed_docs = compressor.compress_documents(unique_docs, query)
                return compressed_docs
            except Exception as e:
                st.warning(f"ドキュメント圧縮中にエラーが発生しました: {str(e)}")
                return unique_docs
        
        return unique_docs
    except Exception as e:
        st.warning(f"ドキュメント処理中にエラーが発生しました: {str(e)}")
        return documents  # エラーが発生した場合は元のドキュメントを返す

# 高度な検索を使用してドキュメント取得
def retrieve_documents(vector_store, query, top_k=4):
    try:
        # 検索クエリのデバッグ情報
        st.sidebar.info(f"検索クエリ: {query}")
        
        # 基本的なリトリーバー
        base_retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",  # より高度な検索タイプを使用
            search_kwargs={
                "k": top_k * 3,  # 再ランキングのためにより多くのドキュメントを取得
                "score_threshold": 0.2  # 類似度のしきい値を下げて多様なドキュメントを取得
            }
        )
        
        # 高度な検索と再ランキングを使用する場合
        if st.session_state.use_advanced_search:
            # より多くのドキュメントを取得
            documents = base_retriever.get_relevant_documents(query)
            
            # 検索結果のデバッグ情報
            with st.sidebar.expander(f"検索結果 ({len(documents)}件)", expanded=False):
                for i, doc in enumerate(documents[:5]):
                    st.write(f"結果 {i+1}:")
                    st.write(f"- スコア: {getattr(doc, 'score', '不明')}")
                    st.write(f"- ソース: {doc.metadata.get('source', '不明')}")
                    st.write(f"- 内容: {doc.page_content[:150]}...")
                if len(documents) > 5:
                    st.write(f"...他 {len(documents) - 5} 件")
            
            # 再ランキングを適用
            if st.session_state.use_document_reranking and documents:
                documents = rerank_documents(documents, query, top_k)
            
            # ドキュメント圧縮を適用
            if documents:
                documents = compress_documents(documents, query)
            
            # 結果数を制限
            return documents[:top_k] if documents else []
        else:
            # よりシンプルだが効果的な検索を使用
            # MMRを使用して多様性のある結果を取得
            mmr_retriever = vector_store.as_retriever(
                search_type="mmr",  # Maximum Marginal Relevance
                search_kwargs={
                    "k": top_k * 2,
                    "fetch_k": top_k * 3,  # 候補を増やす
                    "lambda_mult": 0.7  # 0.7は関連性と多様性のバランス
                }
            )
            
            # 多様性のある検索結果を取得
            documents = mmr_retriever.get_relevant_documents(query)
            
            # 検索結果のデバッグ情報
            with st.sidebar.expander(f"検索結果 ({len(documents)}件)", expanded=False):
                for i, doc in enumerate(documents[:5]):
                    st.write(f"結果 {i+1}:")
                    st.write(f"- ソース: {doc.metadata.get('source', '不明')}")
                    st.write(f"- 内容: {doc.page_content[:150]}...")  # より多くのプレビューを表示
                if len(documents) > 5:
                    st.write(f"...他 {len(documents) - 5} 件")
                    
            # ドキュメント圧縮を適用（標準モードでも適用）
            if documents and st.session_state.use_document_compression:
                documents = compress_documents(documents, query)
                    
            return documents[:top_k]  # 最終的な結果数を制限
            
    except Exception as e:
        st.error(f"ドキュメント検索中にエラーが発生しました: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return []

# チャットボット応答を生成（会話履歴を考慮した対話式）
def generate_conversational_response(user_input, stream_handler=None):
    vector_store = init_vector_db()
    if vector_store is not None:
        try:
            # 処理開始時間
            start_time = time.time()
            
            # 会話メモリの作成または取得
            if 'qa_memory' not in st.session_state:
                st.session_state.qa_memory = ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True,
                    output_key="answer"  # 出力キーを明示的に指定
                )
                
            # 会話履歴からメモリを更新
            if len(st.session_state.conversation_history) > 0 and st.session_state.qa_memory.chat_memory.messages == []:
                for message in st.session_state.conversation_history:
                    if message["role"] == "user":
                        st.session_state.qa_memory.chat_memory.add_user_message(message["content"])
                    elif message["role"] == "assistant":
                        st.session_state.qa_memory.chat_memory.add_ai_message(message["content"])
            
            # プロンプトテンプレートの改良 - より明確な指示を含める
            condense_question_prompt = PromptTemplate.from_template("""
あなたは会話の歴史と最新の質問から、ユーザーの質問の意図を明確にする専門家です。
ユーザーの質問が曖昧な場合は、元の質問を詳細に理解するように再構成してください。
質問が明確であれば、そのまま返してください。

チャット履歴: 
{chat_history}

最新の質問: {question}

明確化された質問（対象物件やチェックイン時間など具体的な情報を含む）:""")

            # QAプロンプトの改良 - コンテキスト利用の明確な指示
            qa_prompt = PromptTemplate.from_template("""
あなたは民泊物件に関する質問に答える専門家アシスタントです。
以下の抽出されたドキュメントの内容と、チャット履歴を使って質問に答えてください。
ドキュメントにない情報については、その旨を明確に伝え、情報を作り上げないでください。

質問内容: {question}

チャット履歴: {chat_history}

ドキュメントから抽出した情報:
{context}

回答のガイドライン:
1. 質問に直接関連する情報のみを回答すること
2. 質問されていない追加情報は含めないこと
3. ドキュメントに記載されている情報のみを使用すること
4. 記載がない情報については「その情報は資料に記載されていません」と明示すること
5. 理解しやすく、丁寧な日本語で回答すること
6. 回答は簡潔に、質問された内容に焦点を当てること
7. 「他に質問があればお知らせください」などの一般的なフレーズは不要です

回答:""")
            
            # コールバックハンドラのリスト
            callbacks = [stream_handler] if stream_handler else None
            
            # モデル選択（高速モードまたは高精度モード）
            model_name = "gpt-3.5-turbo" if st.session_state.use_faster_model else "gpt-4o"
            
            # 質問の曖昧さを判定するモデル - 常に高速モデルを使用
            condense_question_llm = ChatOpenAI(
                temperature=0,
                openai_api_key=OPENAI_API_KEY,
                model_name="gpt-3.5-turbo",  # 質問の要約は常に高速モデルで
                streaming=True if stream_handler else False,
                callbacks=callbacks
            )
            
            # 最終的な回答を生成するモデル
            qa_llm = ChatOpenAI(
                temperature=0,
                openai_api_key=OPENAI_API_KEY,
                model_name=model_name,
                streaming=True if stream_handler else False,
                callbacks=callbacks
            )
            
            # 高度な検索を使用するかどうかのフラグ
            use_advanced = st.session_state.use_advanced_search
            
            try:
                # 高度な検索を使用してドキュメントを取得
                relevant_documents = retrieve_documents(
                    vector_store, 
                    user_input, 
                    top_k=st.session_state.top_k_documents
                )
                
                # 検索結果のデバッグ（サイドバーではなく別の場所に表示も可能）
                if relevant_documents:
                    with st.sidebar.expander("検索結果の詳細情報", expanded=False):
                        st.write(f"取得したドキュメント数: {len(relevant_documents)}")
                        for i, doc in enumerate(relevant_documents[:3]):
                            st.write(f"**ドキュメント {i+1}**:")
                            st.write(f"ソース: {doc.metadata.get('source', '不明')}")
                            st.write(f"内容プレビュー: {doc.page_content[:200]}...")
                else:
                    st.sidebar.warning("検索結果が見つかりませんでした。質問の言い方を変えてみてください。")
            except Exception as e:
                st.error(f"検索エラー: {str(e)}")
                # エラーが発生した場合は高度な検索をオフにして続行
                use_advanced = False
                relevant_documents = []
            
            # 通常のリトリーバー
            retriever = vector_store.as_retriever(
                search_type="mmr",  # MMRでより多様な結果を取得
                search_kwargs={
                    "k": st.session_state.top_k_documents,
                    "fetch_k": st.session_state.top_k_documents * 2,
                    "lambda_mult": 0.7
                }
            )
            
            # ConversationalRetrievalChainの作成
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=qa_llm,
                retriever=retriever,
                memory=st.session_state.qa_memory,
                condense_question_llm=condense_question_llm,
                condense_question_prompt=condense_question_prompt,
                combine_docs_chain_kwargs={"prompt": qa_prompt},
                return_source_documents=True,
                verbose=True,
                output_key="answer"  # 出力キーを明示的に指定
            )
            
            # 高度な検索を使用している場合、取得したドキュメントを使用
            if use_advanced and relevant_documents:
                try:
                    # メモリから会話履歴を取得
                    chat_history = []
                    for msg in st.session_state.qa_memory.chat_memory.messages:
                        if isinstance(msg, HumanMessage):
                            chat_history.append({"type": "human", "content": msg.content})
                        elif isinstance(msg, AIMessage):
                            chat_history.append({"type": "ai", "content": msg.content})
                    
                    # 直接関連ドキュメントを使用
                    result = {
                        "question": user_input,
                        "answer": qa_chain.combine_docs_chain.run(
                            input_documents=relevant_documents,
                            question=user_input,
                            chat_history=chat_history
                        ),
                        "source_documents": relevant_documents
                    }
                except Exception as e:
                    st.error(f"高度な回答生成中にエラーが発生しました: {str(e)}")
                    # エラーが発生した場合は通常の方法で回答
                    result = qa_chain({"question": user_input})
            else:
                # 通常の方法で回答を生成
                result = qa_chain({"question": user_input})
            
            # 処理時間の計算
            processing_time = time.time() - start_time
            
            # 回答文の前処理（重複の検出と除去）
            if 'answer' in result:
                # 回答テキストを取得
                answer_text = result['answer']
                
                # デバッグ用: 元の回答を保存
                original_answer = answer_text
                
                # 重複した文章のパターンを検出して除去
                processed_answer = remove_duplicated_sentences(answer_text)
                
                # デバッグ用: フィルタリング前の回答を保存
                after_dedup = processed_answer
                
                # 不要な情報のフィルタリング
                # フィルタリングを一時的に無効化し、元の回答をそのまま使用
                # processed_answer = filter_irrelevant_information(processed_answer, user_input)
                
                # 処理済みの回答を設定
                result['answer'] = processed_answer
                
                # デバッグ情報を追加（テスト段階では有効化）
                # result['answer'] += "\n\n*デバッグ情報*\n元の回答長: " + str(len(original_answer)) + "\n重複除去後の長さ: " + str(len(after_dedup))
            
            # ソース情報を詳細に追加
            if 'source_documents' in result and result['source_documents']:
                # 回答本文とソース情報を分離
                answer_text = result['answer']
                
                # ソース情報のセクションを作成（よりコンパクトに）
                source_info = "\n\n## 情報源\n"
                
                # 使用されたソースドキュメントを表示
                source_documents = result['source_documents']
                
                # ソース情報の詳細化（簡素化バージョン）
                unique_sources = {}  # ファイル名をキーにしたディクショナリ
                
                # まずメタデータからファイル名を取得
                for i, doc in enumerate(source_documents):
                    if hasattr(doc, 'metadata'):
                        # ソース名を取得
                        source_name = doc.metadata.get('source', f"ソース {i+1}")
                        
                        # ファイル名を適切に抽出
                        if isinstance(source_name, str):
                            # パスからファイル名部分を抽出
                            if '/' in source_name or '\\' in source_name:
                                file_name = os.path.basename(source_name)
                            else:
                                file_name = source_name
                                
                            # ファイルIDプレフィックスがある場合は削除（例：local_ファイル名_日時）
                            if file_name.startswith('local_') and '_' in file_name:
                                parts = file_name.split('_', 2)  # 最初の_でスプリット
                                if len(parts) > 2:
                                    # local_と日時部分を削除
                                    file_name = parts[1]  # 中間部分（実際のファイル名）を取得
                        else:
                            file_name = f"ファイル {i+1}"
                        
                        # 関連性スコア
                        score = getattr(doc, 'score', None)
                        
                        # 既に追加されているソースか確認
                        if file_name not in unique_sources:
                            unique_sources[file_name] = {
                                'name': file_name,
                                'score': score,
                                'count': 1
                            }
                        else:
                            unique_sources[file_name]['count'] += 1
                            # より高いスコアがあれば更新
                            if score and (unique_sources[file_name]['score'] is None or score > unique_sources[file_name]['score']):
                                unique_sources[file_name]['score'] = score
                
                # ソースをスコア順にソート（スコアがある場合）
                sorted_sources = sorted(
                    unique_sources.values(), 
                    key=lambda x: (-(x['score'] or 0), x['name'])  # スコア降順、次にファイル名昇順
                )
                
                # 上位3件のみ表示（簡略化）
                display_sources = sorted_sources[:3]
                
                # ソース情報をシンプルに表示
                for i, source in enumerate(display_sources):
                    source_info += f"{i+1}. {source['name']}\n"
                
                # 全件表示しない場合の注記
                if len(sorted_sources) > 3:
                    source_info += f"他 {len(sorted_sources) - 3} 件のファイル\n"
                
                # 簡易的なコンテキスト情報は表示しない
                # 最終的な回答を生成
                result['answer'] = answer_text + source_info
            
            # 処理時間の追加（簡略化）
            result['answer'] += f"\n\n*処理時間: {processing_time:.2f}秒*"
            
            return result['answer']
        except Exception as e:
            return f"回答の生成中にエラーが発生しました: {str(e)}"
    else:
        return "ベクトルDBが初期化されていません。まずPDFファイルをアップロードしてください。"

# 重複した文章を検出して除去する関数
def remove_duplicated_sentences(text):
    if not text:
        return text
    
    # 文章を分割（句点「。」で区切る）
    sentences = text.split("。")
    
    # 空の文を除外
    sentences = [s.strip() + "。" for s in sentences if s.strip()]
    if not sentences:
        return text
    
    # 最後の文は句点がない場合があるので調整
    if not text.endswith("。"):
        sentences[-1] = sentences[-1][:-1]
    
    # 重複検出と除去
    unique_sentences = []
    seen_sentences = set()
    
    for sentence in sentences:
        # 類似した文章を検出するための正規化
        normalized = sentence.strip().lower()
        
        # 短すぎる文は常に保持（挨拶など）
        if len(normalized) < 10:
            unique_sentences.append(sentence)
            continue
        
        # 既に同じ文があるかチェック
        if normalized not in seen_sentences:
            seen_sentences.add(normalized)
            unique_sentences.append(sentence)
    
    # 重複除去後の文章を再構成
    result = "".join(unique_sentences)
    
    # 連続する句点を単一の句点に置換
    result = result.replace("。。", "。")
    
    return result

# タブ2: Googleドライブからアップロード
def render_google_drive_tab():
    st.subheader("Googleドライブからアップロード")
    
    # Googleドライブ接続ボタン
    if st.button("Googleドライブに接続"):
        with st.spinner("Googleドライブに接続中..."):
            service = authenticate_google_drive()
            if service:
                st.session_state['drive_service'] = service
                st.success("Googleドライブに接続しました！")
            else:
                st.error("Googleドライブへの接続に失敗しました。")
    
    # Googleドライブが接続されている場合
    if 'drive_service' in st.session_state:
        try:
            # フォルダ選択のセクション
            folders = get_drive_folders(st.session_state['drive_service'])
            folder_options = [{"name": "ルートフォルダ", "id": "root"}] + folders
            folder_names = [f"{folder['name']}" for folder in folder_options]
            selected_folder_name = st.selectbox("フォルダを選択", folder_names)
            
            # 選択されたフォルダIDを取得
            selected_folder = next((folder for folder in folder_options if folder["name"] == selected_folder_name), None)
            
            if selected_folder:
                selected_folder_id = selected_folder["id"]
                
                # フォルダ内のすべてのPDFを一括処理するボタン
                if st.button("選択したフォルダ内のすべてのファイルを処理"):
                    with st.spinner("フォルダ内のファイルを一括処理中..."):
                        processed_count = batch_process_drive_files(
                            st.session_state['drive_service'], 
                            selected_folder_id if selected_folder_id != "root" else None
                        )
                        if processed_count > 0:
                            st.success(f"{processed_count}個のファイルを処理しました。")
                            # システムメッセージとして追加
                            st.session_state.messages.append({"role": "system", "content": f"{processed_count}個のファイルを処理しました。"})
                
                # 個別ファイル選択セクション
                st.subheader("または、個別のファイルを選択")
                
                # PDFとマークダウンの両方を取得
                files = get_drive_files(
                    st.session_state['drive_service'], 
                    selected_folder_id if selected_folder_id != "root" else None,
                    file_types=[
                        'application/pdf',        # PDF
                        'text/markdown',          # Markdown
                        'text/x-markdown',        # 別のMD MIMEタイプ
                        'text/plain',             # プレーンテキスト（.mdファイルが時々このタイプに）
                        'application/octet-stream' # バイナリデータ（一部のドライブではこれに）
                    ]
                )
                
                # マークダウンファイルの拡張子でフィルタリング
                markdown_extensions = ['.md', '.markdown', '.mdown', '.mkdn', '.mdx']
                filtered_files = []
                
                for file in files:
                    # PDFファイルはそのまま追加
                    if file['mimeType'] == 'application/pdf':
                        filtered_files.append(file)
                    # ファイル名で.mdなどのマークダウンファイルをチェック
                    elif any(file['name'].lower().endswith(ext) for ext in markdown_extensions):
                        filtered_files.append(file)
                    # "text/markdown" または "text/x-markdown" MIMEタイプならそのまま追加
                    elif file['mimeType'] in ['text/markdown', 'text/x-markdown']:
                        filtered_files.append(file)
                
                if filtered_files:
                    file_names = [f"{file['name']} ({file['mimeType']})" for file in filtered_files]
                    selected_file_name = st.selectbox("ファイルを選択", file_names)
                    
                    # 選択されたファイル名からファイル情報を検索
                    selected_file = None
                    for file in filtered_files:
                        if f"{file['name']} ({file['mimeType']})" == selected_file_name:
                            selected_file = file
                            break
                    
                    if selected_file and st.button("選択したファイルを処理"):
                        with st.spinner("ファイルをダウンロード中..."):
                            file_id = selected_file["id"]
                            file_content = download_file(st.session_state['drive_service'], file_id)
                            
                            if file_content:
                                with st.spinner('ファイルを処理中...'):
                                    # ファイルの種類を判断
                                    file_type = "pdf"  # デフォルト
                                    file_name = selected_file["name"]
                                    
                                    # マークダウンファイルかチェック
                                    if selected_file['mimeType'] in ['text/markdown', 'text/x-markdown'] or any(file_name.lower().endswith(ext) for ext in markdown_extensions):
                                        file_type = "markdown"
                                        suffix = '.md'
                                    else:
                                        suffix = '.pdf'
                                    
                                    # 一時ファイルとして保存
                                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                                        temp_file.write(file_content.getvalue())
                                        temp_file_path = temp_file.name
                                    
                                    try:
                                        # ファイルを開いて処理
                                        with open(temp_file_path, 'rb') as f:
                                            # ファイルタイプに応じた処理
                                            if file_type == "markdown":
                                                text = process_markdown(io.BytesIO(f.read()))
                                                st.info(f"マークダウンファイルとして処理: {file_name}")
                                            else:  # PDF
                                                text = process_pdf(io.BytesIO(f.read()))
                                                st.info(f"PDFファイルとして処理: {file_name}")
                                        
                                        # 一時ファイルを削除
                                        os.unlink(temp_file_path)
                                        
                                        if text:
                                            vector_store = init_vector_db()
                                            vector_store = process_text(text, vector_store, file_name=file_name, file_type=file_type)
                                            if vector_store:
                                                # 処理済みファイル情報を保存
                                                save_processed_file(file_id, file_name, file_type, "google_drive")
                                                st.success('ファイルの処理が完了しました！')
                                                # システムメッセージとして追加
                                                st.session_state.messages.append({"role": "system", "content": f"ファイル「{file_name}」の処理が完了しました。"})
                                            else:
                                                st.error('ベクトル化に失敗しました。')
                                        else:
                                            st.error('ファイルからテキストを抽出できませんでした。')
                                    except Exception as e:
                                        st.error(f"ファイル処理中にエラーが発生しました: {e}")
                                        if os.path.exists(temp_file_path):
                                            os.unlink(temp_file_path)
                else:
                    st.info("選択したフォルダにPDFまたはマークダウンファイルが見つかりませんでした。")
        except Exception as e:
            st.error(f"Googleドライブのファイル取得中にエラーが発生しました: {e}")
            
# サイドバーUI
def render_sidebar():
    with st.sidebar:
        st.title("設定")
        
        # 高速モードの設定
        st.subheader("LLMモデル設定")
        st.session_state.use_faster_model = st.toggle(
            "高速モードを使用",
            value=st.session_state.use_faster_model,
            help="GPT-3.5-turbo（高速）とGPT-4o（高精度）の切り替え"
        )
        
        # 会話記憶の設定
        st.subheader("会話記憶の設定")
        if 'reset_memory_after_each_chat' not in st.session_state:
            st.session_state.reset_memory_after_each_chat = False
            
        st.session_state.reset_memory_after_each_chat = st.toggle(
            "毎回チャットごとに記憶をリセット",
            value=st.session_state.reset_memory_after_each_chat,
            help="ONにすると、1回の質問応答ごとに過去の会話を忘れます"
        )
        
        # 会話履歴を手動でリセットするボタン
        if st.button("会話履歴をリセット", use_container_width=True):
            # 会話履歴をクリア
            st.session_state.messages = []
            st.session_state.conversation_history = []
            if 'qa_memory' in st.session_state:
                st.session_state.qa_memory.clear()
            st.success("会話履歴をリセットしました！")
        
        # 検索文書数の設定
        st.subheader("検索設定")
        st.session_state.top_k_documents = st.slider(
            "検索する関連文書の数",
            min_value=2,
            max_value=12,
            value=st.session_state.top_k_documents,
            help="数を多くすると情報量が増えますが、処理時間が長くなります"
        )
        
        # 高度な検索設定
        st.subheader("高度な検索設定")
        st.session_state.use_advanced_search = st.toggle(
            "高度な検索を使用",
            value=st.session_state.use_advanced_search,
            help="大量のファイルがある場合、検索精度が向上します"
        )
        
        # 高度な検索が有効な場合のみ表示
        if st.session_state.use_advanced_search:
            # ドキュメント再ランキング
            st.session_state.use_document_reranking = st.toggle(
                "ドキュメント再ランキング",
                value=st.session_state.use_document_reranking,
                help="検索結果の順位を再評価し、より関連性の高い情報を優先します"
            )
            
            # ドキュメント圧縮
            st.session_state.use_document_compression = st.toggle(
                "重複情報の削除",
                value=st.session_state.use_document_compression,
                help="重複情報を削除し、処理の効率化を図ります"
            )
        
        # 区切り線
        st.markdown("---")
        
        # タブを作成
        tab1, tab2, tab3 = st.tabs(["ファイルアップロード", "Google Drive", "読み込み済みファイル"])
        
        # タブ1: ローカルからアップロード
        with tab1:
            st.subheader("ローカルからアップロード")
            uploaded_file = st.file_uploader("ファイルをアップロード", type=['pdf', 'md', 'json'])
            
            # ローカルアップロード処理
            if uploaded_file is not None:
                with st.spinner('ファイルを処理中...'):
                    text, file_type = process_file(uploaded_file)
                    if text:
                        vector_store = init_vector_db()
                        # ファイル名とタイプを渡す
                        vector_store = process_text(text, vector_store, file_name=uploaded_file.name, file_type=file_type)
                        if vector_store:
                            # 処理済みファイル情報を保存
                            file_id = f"local_{uploaded_file.name}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
                            save_processed_file(file_id, uploaded_file.name, file_type, "local")
                            
                            st.success('ファイルの処理が完了しました！')
                            # システムメッセージとして追加
                            st.session_state.messages.append({"role": "system", "content": f"ファイル「{uploaded_file.name}」の処理が完了しました。"})
                        else:
                            st.error('ファイルの処理中にエラーが発生しました。')
                    else:
                        st.error('ファイルからテキストを抽出できませんでした。')
        
        # タブ2: Googleドライブからアップロード
        with tab2:
            render_google_drive_tab()
        
        # タブ3: 読み込み済みファイル
        with tab3:
            display_processed_files()

# メインアプリケーション
def main():
    st.title("フロント対応チャットボット")
    
    # 認証
    try:
        name, authentication_status, username = authenticator.login(
            fields={'Form name': 'Login', 'Username': 'Username', 'Password': 'Password'}
        )
    except Exception as e:
        st.error(f"認証エラーが発生しました: {str(e)}")
        # エラーが発生した場合、以降の処理に進まないようにする
        authentication_status = None # ログイン失敗扱いにする
    
    if authentication_status:
        authenticator.logout('Logout', 'main')
        st.write(f'ようこそ *{name}*')
        
        # サイドバーの表示
        render_sidebar()
        
        # チャットインターフェース
        
        # 物件ID・質問内容入力エリア
        with st.expander("物件IDと質問テンプレート", expanded=False):
            st.write("物件IDと質問内容を入力すると、自動的にプロンプトが作成されます")
            
            col1, col2 = st.columns([1, 3])
            with col1:
                property_id = st.text_input("物件ID", key="property_id", placeholder="例: P001")
            with col2:
                question_type = st.selectbox(
                    "質問タイプ",
                    [
                        "選択してください...",
                        "Wi-Fi情報について教えてください",
                        "チェックイン方法について教えてください", 
                        "チェックアウト時間について教えてください",
                        "最寄りの交通機関について教えてください",
                        "駐車場の利用方法について教えてください",
                        "周辺の飲食店について教えてください",
                        "追加料金について教えてください",
                        "設備・アメニティについて教えてください"
                    ]
                )
            
            # 質問内容の手動入力
            custom_question = st.text_input("カスタム質問（自由入力）", key="custom_question", placeholder="ここに質問を入力できます")
            
            # プロンプト生成ボタン
            if st.button("プロンプトを生成", use_container_width=True):
                if property_id:
                    # プロンプトの生成
                    if custom_question:
                        # カスタム質問が入力されている場合はそれを使用
                        prompt = f"物件ID {property_id} について。{custom_question}"
                    elif question_type != "選択してください...":
                        # 質問タイプが選択されている場合はそれを使用
                        prompt = f"物件ID {property_id} について。{question_type}"
                    else:
                        # どちらも指定されていない場合
                        prompt = f"物件ID {property_id} の詳細情報を教えてください。"
                    
                    # 生成したプロンプトをセッションに保存
                    st.session_state.generated_prompt = prompt
                    st.success(f"プロンプトを生成しました: {prompt}")
                    
                    # 質問として追加
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    st.session_state.conversation_history.append({"role": "user", "content": prompt})
                    
                    # チャットの応答を生成
                    with st.chat_message("assistant"):
                        response_container = st.empty()
                        stream_handler = StreamHandler(response_container)
                        
                        with st.spinner("回答を生成中..."):
                            # 会話履歴を考慮した応答生成 (ストリーミング対応)
                            response = generate_conversational_response(prompt, stream_handler)
                            
                            # ボットの応答をチャット履歴に追加
                            st.session_state.messages.append({"role": "assistant", "content": response})
                            st.session_state.conversation_history.append({"role": "assistant", "content": response})
                    
                    # 画面を再読み込み
                    st.rerun()
                else:
                    st.error("物件IDを入力してください")
        
        # デバッグパネル（プロンプトと生成情報の表示）
        with st.expander("デバッグ情報", expanded=False):
            st.subheader("システム情報")
            # model_nameとsearch_modeを直接ここで計算して使用
            debug_model_name = "GPT-3.5-turbo (高速モード)" if st.session_state.use_faster_model else "GPT-4o (高精度モード)"
            debug_search_mode = "高度な検索" if st.session_state.use_advanced_search else "通常検索"
            
            st.write(f"使用中のモデル: {debug_model_name}")
            st.write(f"検索モード: {debug_search_mode}")
            st.write(f"検索文書数: {st.session_state.top_k_documents}件")
            
            st.subheader("最後に生成されたプロンプト")
            if 'generated_prompt' in st.session_state:
                st.code(st.session_state.generated_prompt)
            else:
                st.write("まだプロンプトが生成されていません")
            
            st.subheader("最後の質問と回答")
            if len(st.session_state.conversation_history) >= 2:
                last_question = next((msg["content"] for msg in reversed(st.session_state.conversation_history) if msg["role"] == "user"), "なし")
                st.write("**最後の質問:**")
                st.write(last_question)
                
                last_answer = next((msg["content"] for msg in reversed(st.session_state.conversation_history) if msg["role"] == "assistant"), "なし")
                st.write("**最後の回答:**")
                st.write(last_answer)
            else:
                st.write("まだ会話履歴がありません")
        
        # パフォーマンスモード表示
        model_name = "GPT-3.5-turbo (高速モード)" if st.session_state.use_faster_model else "GPT-4o (高精度モード)"
        search_mode = "高度な検索" if st.session_state.use_advanced_search else "通常検索"
        st.caption(f"現在のモデル: {model_name} / 検索モード: {search_mode} / 検索文書数: {st.session_state.top_k_documents}件")
        
        # チャット履歴の表示
        for message in st.session_state.messages:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            elif message["role"] == "assistant":
                with st.chat_message("assistant"):
                    st.write(message["content"])
            elif message["role"] == "system":
                with st.chat_message("system"):
                    st.info(message["content"])
        
        # チャット入力
        user_input = st.chat_input("質問を入力してください...")
        
        if user_input:
            # ユーザーの質問をチャット履歴に追加
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.conversation_history.append({"role": "user", "content": user_input})
            
            # ユーザーの質問を表示
            with st.chat_message("user"):
                st.write(user_input)
            
            # ボットの応答を生成 (ストリーミング対応)
            with st.chat_message("assistant"):
                response_container = st.empty()
                stream_handler = StreamHandler(response_container)
                
                with st.spinner("回答を生成中..."):
                    # 会話履歴を考慮した応答生成 (ストリーミング対応)
                    response = generate_conversational_response(user_input, stream_handler)
                    
                    # ボットの応答をチャット履歴に追加
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.session_state.conversation_history.append({"role": "assistant", "content": response})
                
                # 毎回チャットごとに記憶をリセットする設定がオンの場合
                if st.session_state.reset_memory_after_each_chat:
                    # 最新の会話だけを残して他をクリア
                    latest_user_msg = {"role": "user", "content": user_input}
                    latest_assistant_msg = {"role": "assistant", "content": response}
                    
                    # 会話履歴をリセット（最新の1往復だけ残す）
                    st.session_state.conversation_history = [latest_user_msg, latest_assistant_msg]
                    
                    # LLMのメモリもクリア
                    if 'qa_memory' in st.session_state:
                        st.session_state.qa_memory.clear()
                        # 最新の1往復だけメモリに追加
                        st.session_state.qa_memory.chat_memory.add_user_message(user_input)
                        st.session_state.qa_memory.chat_memory.add_ai_message(response)
                    
                    # UIには全会話履歴を表示したまま（リセットしない）
                    # st.session_state.messages は変更しない
                    
                    # デバッグメッセージ（必要に応じて）
                    # st.session_state.messages.append({"role": "system", "content": "会話記憶をリセットしました"})
    
    elif authentication_status == False:
        st.error('ユーザー名またはパスワードが間違っています')
    elif authentication_status == None:
        # ログインフォーム表示中のため、ここでは何もしないか、
        # 必要であれば「ログインしてください」のメッセージを出す
        pass

# 質問に関係ない情報をフィルタリングする関数
def filter_irrelevant_information(answer, question):
    """
    質問に直接関連しない情報をフィルタリングする
    
    Args:
        answer (str): 元の回答テキスト
        question (str): ユーザーの質問
    
    Returns:
        str: フィルタリング後の回答テキスト
    """
    # 質問内容が空の場合はフィルタリングしない
    if not question or not answer:
        return answer
        
    # 質問に関連するキーワードを抽出
    question_lower = question.lower()
    key_topics = []
    
    # 質問のタイプを判定するキーワード
    if 'wi-fi' in question_lower or 'wifi' in question_lower or 'インターネット' in question_lower:
        key_topics.append('wifi')
    if 'チェックイン' in question_lower or '到着' in question_lower:
        key_topics.append('checkin')
    if 'チェックアウト' in question_lower or '出発' in question_lower or '退出' in question_lower:
        key_topics.append('checkout')
    if '料金' in question_lower or '追加料金' in question_lower or '費用' in question_lower:
        key_topics.append('price')
    if '駐車' in question_lower or '駐車場' in question_lower:
        key_topics.append('parking')
    if '場所' in question_lower or '地図' in question_lower or 'アクセス' in question_lower:
        key_topics.append('location')
    if '設備' in question_lower or 'アメニティ' in question_lower:
        key_topics.append('amenity')
    
    # キーワードが検出されない場合はフィルタリングしない
    if not key_topics:
        return answer
    
    # 文単位で分割
    sentences = []
    current = ""
    for char in answer:
        current += char
        if char in ['。', '！', '？', '!', '?', '.'] and current.strip():
            sentences.append(current.strip())
            current = ""
    if current.strip():
        sentences.append(current.strip())
    
    # 各トピックに関連するキーワード（含めるべき単語リスト）
    included_keywords = {
        'wifi': ['Wi-Fi', 'WIFI', 'wifi', 'インターネット', 'ネット', 'SSID', 'パスワード', 'ルーター', '接続', 'ワイファイ'],
        'checkin': ['チェックイン', '到着', '入室', '鍵', '受付', 'チェックイン時間'],
        'checkout': ['チェックアウト', '退室', '出発', 'チェックアウト時間'],
        'price': ['料金', '費用', '代金', '追加料金', '円', '¥', '金額'],
        'parking': ['駐車', '駐車場', '車', '停める', 'パーキング'],
        'location': ['住所', '場所', 'アクセス', '地図', '位置', '最寄り'],
        'amenity': ['設備', 'アメニティ', '備品', '用意', 'シャンプー', 'タオル']
    }
    
    # チェックインの不要な情報を検出する排除キーワード
    excluded_checkin_phrases = [
        'チェックイン時間', 'チェックイン予定', 'チェックインは', 'チェックインについて', 
        '入室時間', '到着時間', '鍵の受け取り'
    ]
    
    # チェックアウトの不要な情報を検出する排除キーワード
    excluded_checkout_phrases = [
        'チェックアウト時間', 'チェックアウト予定', 'チェックアウトは', 'チェックアウトについて',
        '退室時間', '出発時間'
    ]
    
    # 質問に関係ない情報を含む文を除外する
    filtered_sentences = []
    
    for sentence in sentences:
        # デフォルトでは文を含める
        include_sentence = False
        
        # 質問に関連するトピックのキーワードを含む文は含める
        for topic in key_topics:
            if any(keyword in sentence for keyword in included_keywords.get(topic, [])):
                include_sentence = True
                break
        
        # 一般的な回答の冒頭部分は常に含める
        if '以下の通りです' in sentence or '物件ID' in sentence:
            include_sentence = True
        
        # 「〜については資料に記載されていません」という文で、
        # 質問していない内容に関する記述は除外
        if 'wifi' not in key_topics and any(keyword in sentence for keyword in included_keywords.get('wifi', [])):
            include_sentence = False
        
        if 'checkin' not in key_topics and any(phrase in sentence for phrase in excluded_checkin_phrases):
            include_sentence = False
        
        if 'checkout' not in key_topics and any(phrase in sentence for phrase in excluded_checkout_phrases):
            include_sentence = False
        
        if include_sentence:
            filtered_sentences.append(sentence)
    
    # フィルタリングの結果、文がなくなった場合は元の回答を返す
    if not filtered_sentences:
        return answer
    
    # 文を結合して返す
    filtered_answer = ''.join(filtered_sentences)
    return filtered_answer

if __name__ == "__main__":
    # .envファイルが存在するか確認
    if not os.path.exists(".env"):
        st.error(".envファイルが見つかりません。アプリケーションを実行するディレクトリに配置してください。")
        st.stop()
    main() 