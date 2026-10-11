"""Builds executed notebooks that load the saved results and re-run the cheap correctness checks."""
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

def nb(name, title, cells):
    n = nbf.v4.new_notebook()
    n.cells = [nbf.v4.new_markdown_cell(f"# {title}")] + [
        (nbf.v4.new_code_cell(c) if k == "code" else nbf.v4.new_markdown_cell(c)) for k, c in cells]
    ExecutePreprocessor(timeout=600, kernel_name="python3").preprocess(n, {"metadata": {"path": "."}})
    nbf.write(n, f"notebooks/{name}.ipynb")

HEAD = "import pandas as pd, json, glob\nfrom IPython.display import Image, display\nR='results/%s/'\n"
nb("scratch_gcn", "Scratch GCN", [
 ("code", "import torch\nfrom scratch_gcn.core import *\nset_seed(0)\nx,y,ei,m=make_dataset(seed=0)\nmodel=GCN(3,16,2); train_model(model,x,y,prep_graph(ei,200),m)\nprint('manual layer check', check_manual(model,x,ei,node=0))\nprint('permutation check', check_permutation(model,x,ei))"),
 ("code", HEAD % "scratch_gcn" + "display(pd.read_csv(R+'ablation_results.csv'));display(pd.read_csv(R+'homophily_sweep.csv'));display(Image(R+'structure_evidence.png'))"),
 ("md", "Dataset design and interpretation: see `report/TECHNICAL_REPORT.md` section 1.")])
nb("cora_benchmark", "Cora benchmark", [
 ("code", HEAD % "cora_benchmark" + "print(json.load(open(R+'dataset_inspection.json')));display(pd.read_csv(R+'comparison.csv'));display(Image(R+'depth_and_edge_studies.png'))"),
 ("code", "print(open(R+'error_analysis.md').read())"),
 ("md", "Interpretation and recommendation: see report section 2.")])
nb("link_prediction", "Link prediction and leakage", [
 ("code", HEAD % "link_prediction" + "print(json.load(open(R+'leakage_checks.json')));display(pd.read_csv(R+'results_valid_vs_leaky.csv'));display(pd.read_csv(R+'top10_candidate_links.csv'));display(pd.read_csv(R+'failure_cases.csv'));display(Image(R+'valid_vs_leaky.png'))"),
 ("md", "Why leaky evaluation is invalid and why unobserved links are not confirmed negatives: report section 3.")])
nb("mlops", "MLOps pipeline", [
 ("code", "R='results/mlops/'\nprint(open(R+'demo_log.txt').read())"),
 ("code", "print(open(R+'failure_handling.md').read())\nimport json;print(json.dumps(json.load(open(R+'drift_demo.json')),indent=1)[:3000])"),
 ("md", "Live demo commands: see README.")])
