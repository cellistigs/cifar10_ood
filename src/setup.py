import setuptools
import os

loc = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(loc,"../","README.md"),"r") as fh:
    long_description = fh.read()

setuptools.setup(
        name = "cifar10_ood",
        version = "0.0.1",
        author = "Taiga Abe",
        author_email = "ta2507@columbia.edu",
        description = "OOD dataset for cifar10 inputorch", 
        long_description = long_description,
        long_description_content_type = "test/markdown", 
        url = "https://github.com/cellistigs/cifar10_ood",
        packages = setuptools.find_packages(),
        include_package_data=True,
        package_data={},
        classifiers = [
            "License :: OSI Approved :: MIT License"],
        python_requires=">=3.6",
        )

