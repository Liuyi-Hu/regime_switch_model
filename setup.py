from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

install_requires = ["numpy", "scikit-learn", "scipy"]

classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering',
      ]

setup_options = dict(
    name='regime_switch_model',
    description='Regime-Switching Model',
    long_description=readme(),
    version='0.1.1',
    url='https://github.com/Liuyi-Hu/regime_switch_model',
    author='Liuyi Hu',
    author_email='liuyi.hu.apply@gmail.com',
    license='new BSD',
    packages=['regime_switch_model'],
    classifiers=classifiers,
    install_requires=install_requires
)


if __name__ == "__main__":
    setup(**setup_options)
