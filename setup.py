"""
-- Created by: Ashok Kumar Pant
-- Created on: 2/25/20
"""
from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = []
try:
    with open('requirements.txt') as f:
        requirements = f.read().splitlines()
except IOError as e:
    print(e)
setup(
    name='yolov5',
    version='1.0',
    description="",
    long_description=readme + '\n',
    author="Ashok Kumar Pant",
    author_email='ashokpant@treeleaf.ai',
    url='',
    packages=find_packages(),
    package_dir={},
    package_data={},
    install_requires=requirements,
    license="",
    zip_safe=False,
    keywords=''
)
