import setuptools


description = """Pytorch version of resnet50 and resnet200 checkpoints of
"Efficient Visual Pretraining with Contrastive Detection" model"""


setuptools.setup(
    name="pytorch_eff_vis_pretraining/download.py",
    version="1.0.0",
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
    package_dir={"": "pytorch_eff_vis_pretraining"},
    packages=[
            'torch',
            'tqdm',
        ],
    python_requires=">=3.8",
)
