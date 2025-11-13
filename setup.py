from setuptools import find_packages, setup

setup(
    name="prepview_engine",
    version="0.0.1",
    author="InshalMunaf", 
    author_email="inshalmunaf@gmail.com", 
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)