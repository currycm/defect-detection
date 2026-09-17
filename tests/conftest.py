"""将项目根目录加入 sys.path，便于导入 src 包。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
