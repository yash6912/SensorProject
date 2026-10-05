from setuptools import setup, find_packages
from typing import List

def get_requirements(file_path: str) -> List[str]:
    requirements = [] 
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        if "-e ." in requirements:
            requirements.remove("-e .")
    return requirements

setup(
    name="fault_detection",
    version="0.0.1",
    author="Yash",
    author_email="<yashkanoje117@gmail.com>",
    install_requirements=get_requirements('requirements.txt'),
    packages=find_packages()
)