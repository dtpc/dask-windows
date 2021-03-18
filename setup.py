from setuptools import find_packages, setup

tests_require = [
    "black>=20.8b1",
    "flake8>=3.8.0",
    "flake8-isort>=4.0.0",
    "isort>=5.1.0",
    "mypy",
    "pytest",
    "pytest-cov",
]


setup(
    name="dask_windows",
    author="Dave Cole",
    maintainer="Dave Cole",
    license="Apache License 2.0",
    description="Windowed view into N-d dask array",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.6.0",
    packages=find_packages(exclude=("tests", "tests.*")),
    use_scm_version={
        "write_to": "dask_windows/_version.py",
    },
    setup_requires=["setuptools_scm"],
    install_requires=[
        "scikit-image",
        "numpy",
        "dask[array]",
    ],
    extras_require={"test": tests_require},
    tests_require=tests_require,
)
