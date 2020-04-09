import setuptools

setuptools.setup(
    name="random_lumberjacks",
    version="0.0.1",
    description="A collection of handy data science classes and functions.",
    packages=setuptools.find_packages('src'),
    package_dir={'': 'src'}
)