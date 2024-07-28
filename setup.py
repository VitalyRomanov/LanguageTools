from distutils.core import setup
from Cython.Build import cythonize

setup(
      name='LanguageTools',
      version='0.0.1',
      py_modules=['LanguageTools'],
      install_requires=[
            "numpy",
            "gensim",
            "psutil",
            "tqdm",
            "nltk",
            "more_itertools"
      ],
      packages=["LanguageTools"],
      ext_modules=cythonize("LanguageTools/utils/extra_utils.pyx"),
)
