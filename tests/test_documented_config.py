from pathlib import Path
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import unittest

PROJECT=Path(os.environ.get('DATHINKER_SOURCE',Path(__file__).resolve().parents[1]))
MARKER='synthetic-test-not-a-credential'
BOOTSTRAP='''import socket,runpy
attempts=[]
def blocked(*args,**kwargs):
 attempts.append(1)
 raise AssertionError('Network forbidden')
socket.socket.connect=blocked
socket.create_connection=blocked
runpy.run_module('dathinker.cli',run_name='__main__')
assert attempts == [], 'Unexpected network attempt'
'''

class DocumentedConfigTests(unittest.TestCase):
    def run_cli(self, body):
        with tempfile.TemporaryDirectory(prefix='dathinker-startup-')as temp:
            root=Path(temp);home=root/'home';config=home/'.config/dathinker';config.mkdir(parents=True)
            (config/'config.toml').write_text(body)
            env={k:v for k,v in os.environ.items()if k not in ['OPENROUTER_API_KEY','XDG_CONFIG_HOME','DATHINKER_MODEL','DATHINKER_ROUTER']}
            env.update(HOME=str(home),PYTHONPATH=str(PROJECT/'src'),NO_COLOR='1')
            result=subprocess.run([sys.executable,'-c',BOOTSTRAP],cwd=root,env=env,input='\n',text=True,capture_output=True,timeout=15)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertNotIn(MARKER,result.stdout+result.stderr)
            return result.stdout

    def test_exact_preferred_readme_setup_reaches_topic_prompt(self):
        text=(PROJECT/'README.md').read_text()
        line=next(line.strip()for line in text.splitlines()if line.strip().startswith('echo ')and'config.toml'in line)
        tokens=shlex.split(line)
        self.assertEqual(tokens[0],'echo');self.assertEqual(tokens[2],'>')
        # Parse the instruction as data, never execute README shell content.
        body=re.sub(r'"[^"]*"','"'+MARKER+'"',tokens[1])+'\n'
        output=self.run_cli(body)
        self.assertNotIn('OPENROUTER_API_KEY not found',output)
        self.assertIn('No topic provided. Exiting.',output)

    def test_existing_flat_and_nested_formats_still_start(self):
        for body in ['openrouter_api_key = "'+MARKER+'"\n','[api]\nopenrouter_key = "'+MARKER+'"\n']:
            with self.subTest(format=body.split('=')[0]):
                self.assertIn('No topic provided. Exiting.',self.run_cli(body))

    def test_existing_file_and_environment_precedence_is_unchanged(self):
        with tempfile.TemporaryDirectory(prefix='dathinker-precedence-')as temp:
            root=Path(temp);config=root/'dathinker';config.mkdir();(config/'config.toml').write_text('openrouter_api_key = "synthetic-flat"\n[api]\nopenrouter_key = "synthetic-nested"\n')
            env={k:v for k,v in os.environ.items()if k not in ['OPENROUTER_API_KEY','DATHINKER_MODEL','DATHINKER_ROUTER']};env.update(XDG_CONFIG_HOME=str(root),PYTHONPATH=str(PROJECT/'src'))
            program='from dathinker.config import load_config; import os; assert load_config().openrouter_api_key == os.environ["EXPECTED_MARKER"]'
            for expected,extra in [('synthetic-nested',{}),('synthetic-env',{'OPENROUTER_API_KEY':'synthetic-env'})]:
                result=subprocess.run([sys.executable,'-c',program],env={**env,**extra,'EXPECTED_MARKER':expected},cwd=root,text=True,capture_output=True,timeout=10)
                self.assertEqual(result.returncode,0,result.stderr)

if __name__=='__main__':
    unittest.main()
