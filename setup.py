from setuptools import setup, find_packages

setup(
    name='nnls',
    version='0.0.1',
    description="""Nonnegative least squares solvers in Python/Cython""",
    author="Joseph Knox",
    author_email="joseph.edward.knox@gmail.com",
    url='https://github.com/jknox13/python-nnls',
    packages=find_packages(),
    include_package_data=True,
    setup_requires=['pytest-runner'],
)
