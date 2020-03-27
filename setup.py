import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pytorch-chitra-aravinddcsadguru", # Replace with your own username
    version="0.0.4",
    author="Aravind D Chakravarti",
    author_email="aravind.deep.learning@gmail.com",
    description="Deep Learning Library Based on Pytorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aravindchakravarti/pytorch-chitra",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)