from setuptools import setup

setup(name='neuralsolver',
      version='0.0.1',
      description='Solve ODE by neural network',
      url='https://github.com/JiaweiZhuang/AM205_final',
      packages=['neuralsolver'],
      python_requires=">=3.5",
      install_requires=['autograd==1.2', 'scipy==1.0']
      )
