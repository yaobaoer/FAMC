from pathlib import Path
import runpy
import sys
p = Path(__file__).resolve().parents[1] / 'common'
sys.path.insert(0, str(p))
runpy.run_path(str(p / '_run_ablation_swap_lodo.py'), run_name='__main__')
