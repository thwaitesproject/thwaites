from firedrake import Mesh
from os.path import join, abspath, dirname
from pathlib import Path

data_dir = Path(abspath(dirname(__file__))) / 'data'
tmp_dir = Path(abspath(dirname(__file__))) / 'tmp'
tmp_dir.mkdir(exist_ok=True)

def get_coarse_mesh():
    return Mesh(str(data_dir / 'coarse.msh'))
