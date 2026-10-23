"""Merge the official experimental branch, preserving local patch history."""
from pathlib import Path
import json, subprocess

ROOT=Path(__file__).resolve().parent.parent
URL='https://github.com/LostRuins/koboldcpp.git'
BRANCH='concedo_experimental'

def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()

def main():
    if git('status','--porcelain'):raise RuntimeError('Commit or preserve existing source changes first')
    branch=git('branch','--show-current')
    if branch!=BRANCH:raise RuntimeError('Checkout the patched concedo_experimental branch first')
    old=git('rev-parse','HEAD')
    subprocess.run(['git','fetch',URL,'refs/heads/'+BRANCH],cwd=ROOT,check=True)
    official=git('rev-parse','FETCH_HEAD')
    result=subprocess.run(['git','merge','--no-ff','--no-edit',official],cwd=ROOT)
    if result.returncode:raise SystemExit('Resolve merge conflicts by meaning; build/release blocked.')
    print(json.dumps({'before':old,'official_experimental':official,'merged':git('rev-parse','HEAD'),
                      'must_rebuild_retrain_validate':old!=git('rev-parse','HEAD')}))

if __name__=='__main__':main()
