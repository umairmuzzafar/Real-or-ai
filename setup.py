from setuptools import setup
from setuptools.command.install import install
import subprocess
import sys

class CustomInstall(install):
    def run(self):
        # Install system dependencies
        subprocess.check_call(['apt-get', 'update'])
        subprocess.check_call(['apt-get', 'install', '-y', 'build-essential', 'cmake'])
        
        # Install sentencepiece from pre-built wheel
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
                             '--find-links', 'https://pypi.ngc.nvidia.com',
                             'sentencepiece'])
        
        # Install remaining requirements
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])

setup(
    name='real_or_ai',
    version='1.0',
    cmdclass={'install': CustomInstall},
)
