# JapanTaxSimulator における uv 使用ガイド

## 基本的な使い方

### 1. 依存関係のインストール/更新
```bash
# pyproject.tomlから自動でインストール
uv sync
```

### 2. パッケージの追加
```bash
# 新しいパッケージを追加
uv add requests

# 開発用パッケージを追加
uv add --dev black ruff
```

### 3. パッケージの削除
```bash
uv remove requests
```

### 4. コマンドの実行
```bash
# uvの環境で実行（アクティベート不要）
uv run python quick_check.py
uv run jupyter notebook
uv run pytest
```

### 5. 環境のリセット
```bash
# .venvを削除して再作成
rm -rf .venv
uv sync
```

## メリット

- ✅ **高速**: pipより10-100倍速い
- ✅ **自動管理**: 仮想環境の作成・管理が自動
- ✅ **依存関係解決**: より賢い依存関係の解決
- ✅ **ロックファイル**: uv.lockで再現性を保証
- ✅ **アクティベート不要**: `uv run`で直接実行可能

## よく使うコマンド

```bash
# 開発サーバー起動
uv run jupyter notebook

# テスト実行
uv run pytest

# 任意のPythonスクリプト実行
uv run python src/tax_simulator.py

# インタラクティブシェル
uv run python
```

## 従来の方法との対応

| 従来の方法 | uvの方法 |
|-----------|----------|
| `pip install -r requirements.txt` | `uv sync` |
| `pip install package` | `uv add package` |
| `source venv/bin/activate` | 不要（`uv run`を使用） |
| `python script.py` | `uv run python script.py` |