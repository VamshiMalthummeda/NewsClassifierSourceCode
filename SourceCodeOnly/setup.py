import os
from setuptools import find_packages, setup
def get_files():
    files = []
    name_dir = os.path.dirname(os.getcwd())
    files.append(('',['.env']))
    files.append(('templates',['templates/index.html']))
    return files
setup(
    name='newsclassifier',
    packages=find_packages(),
    version='0.1.0',
    description='Classification of BBC News articles',
    author='Vamshi Krishna',
    license='MIT',
    author_email='mvamsikhyd@gmail.com',
    maintainer='Vamshi Krishna',
    maintainer_email='mvamsikhyd@gmail.com',
    url="",
    install_requires = ['requests','python-dotenv','nltk','numpy','sklearn','scipy','flask','yellowbrick'],
    classifiers = [
        'Programming Language:: Python :: 3.6',
        'Programming Language:: Python :: 3.7'
        ],
    keywords = "News classification using TFIDF Vectorizer",
    entry_points={
	'console_scripts': ['FitModel=src.batch:main'],
    },
    include_package_data=True,
    data_files = get_files(),
)
