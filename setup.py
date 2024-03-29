from setuptools import find_packages, setup

setup(
    name="warmup",
    version="0.3.0",
    install_requires=["gymnasium", "pyyaml", "torch", "imageio"],
    author="Pierre Schumacher, MPI-IS Tuebingen, Autonomous Learning",
    author_email="pierre.schumacher@tuebingen.mpg.de",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    package_data={"": ["*.yaml", "*.stl", "*.xml"]},
)
