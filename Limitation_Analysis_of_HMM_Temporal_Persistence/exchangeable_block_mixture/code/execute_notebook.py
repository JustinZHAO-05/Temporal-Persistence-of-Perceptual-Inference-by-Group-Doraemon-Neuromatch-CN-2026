from pathlib import Path
import os,sys
runtime=Path(__file__).resolve().parents[2]/"tmp"/"jupyter-notebook"/"runtime";sys.path.insert(0,str(runtime))
import nbformat
from nbclient import NotebookClient
p=Path(sys.argv[1]).resolve();nb=nbformat.read(p,as_version=4);env=os.environ.copy();env["PYTHONPATH"]=str(runtime)+os.pathsep+env.get("PYTHONPATH","")
NotebookClient(nb,timeout=None,kernel_name="python3",resources={"metadata":{"path":str(p.parents[1])}}).execute(env=env)
nbformat.write(nb,p);print(p)
