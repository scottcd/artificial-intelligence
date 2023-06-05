from setuptools import setup, find_packages

setup(
    name='artificial_intelligence',
    version='1.0.0',
    description='Three AI algorithms',
    author='Chandler Scott',
    author_email='scottcd1@etsu.edu',
    packages=find_packages(),
    install_requires=[
        'torch>=1.8.0',
        'torchvision>=0.9.0',
        'gymnasium[all]',
        'pygame',
        # Add other dependencies if needed
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.11',
    ],
)