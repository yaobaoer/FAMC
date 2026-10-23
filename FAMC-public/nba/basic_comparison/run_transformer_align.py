from pathlib import Path
import runpy
import sys
p = Path(__file__).resolve().parents[1] / 'common'
sys.path.insert(0, str(p))
runpy.run_path(str(p / 'run_transformer_align.py'), run_name='__main__')
