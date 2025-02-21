from setuptools import setup, find_packages

setup(
    name='my_utils',
    version='0.0.0',
    description='A collection of my utils',
    author='Samuel Longo',
    author_email='longo.samuel@outlook.it',
    url='https://github.com/chewingram/my_utils',
    packages=find_packages(),
    include_package_data=True,  # Ensure this is present
    entry_points={
        "console_scripts": [
            "make_md = my_utils.utils_md:make_md",
            "atlen = my_utils.utils:atlen",
            "wrap = my_utils.utils_md:wrap",
            "at_extract = my_utils.utils:at_extract",
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Unix',
    ],
    package_data={"my_utils": ["data/*"]},
)
