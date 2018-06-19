from setuptools import setup



setup(name='Entailment',
      packages=['entailment'],
      install_requires=['tensorflow', 'numpy'],
      dependency_links=['git+https://github.com/act65/logical-entailment-dataset.git@master#egg=led_parser-1.0']
      )
