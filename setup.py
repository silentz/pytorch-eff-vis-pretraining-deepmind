from setuptools import setup

description = """Pytorch version of resnet50 and resnet200 checkpoints of
"Efficient Visual Pretraining with Contrastive Detection" model"""

setup(
    name="pytorch_eff_vis_pretraining",
    version="1.0.1",
    author="Pershin Maxim",
    author_email="mepershin@example.com",
    description=description,
    long_description=description,
    long_description_content_type="text/markdown",
    url="https://github.com/silentz/pytorch-eff-vis-pretraining-deepmind",
    project_urls={
        "Bug Tracker": "https://github.com/silentz/pytorch-eff-vis-pretraining-deepmind/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    packages=['pytorch_eff_vis_pretraining'],
    install_requires=[
       "tqdm",
       "torch>=1.8",
    ],
    python_requires=">=3.8",
)
