from pathlib import Path

from setuptools import find_namespace_packages, setup


ROOT = Path(__file__).parent

setup(
    name="diffupercom",
    version="0.1.0",
    description="Code for DiffuPercom, a simplex-based diffusion framework for personalized comment generation.",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/yahiko-l/PersonalComment",
    project_urls={
        "Paper": "https://doi.org/10.1016/j.asoc.2026.115493",
        "Dataset": "https://github.com/yahiko-l/PersonalComment",
        "Source": "https://github.com/yahiko-l/PersonalComment",
    },
    author="Jiamiao Liu, Pengsen Cheng, Jinqiao Dai, Jiayong Liu",
    keywords="simplex diffusion personalized comment generation PFA",
    license="Apache-2.0",
    packages=find_namespace_packages(include=("sdlm*", "classifier*")),
    python_requires=">=3.8,<3.11",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
