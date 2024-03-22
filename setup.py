from setuptools import setup, find_packages

setup(
    name='HELIOS',
    version='0.1.0',
    author='Matej Malik',
    author_email='',
    description='1D radiatrive transfer in (exo-)planet atmospheres',
    long_description='Long description of my package',
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/my_package',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        # Add other dependencies here
    ],
    package_data={
        "": [
            "",
        ]
    },
)
