from setuptools import setup

packages = [
    'numpy',
    'scipy',
    'matplotlib',
    'sklearn',
    'tensorflow',
    'cvxpy',
    'torch',
    'pymanopt',
    'pandas',
    'mosek',
]
#
setup(name='sensepy',
      version='0.0.2',
      description='Point process sensing library',
      url='',
      author='Mojmir Mutny',
      author_email='mojmir.mutny@inf.ethz.ch',
      license='custom',
      packages=['sensepy'],
	    zip_safe=False,
      install_requires=packages)
