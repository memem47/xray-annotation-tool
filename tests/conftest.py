# tests/conftest.py
import sys
from pathlib import Path

# プロジェクトルート/src を import パスに追加
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))          # ルート自体
sys.path.insert(0, str(ROOT / "src"))  # src/ を直指定（念のため）
