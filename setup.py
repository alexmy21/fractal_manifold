# setup.py - Build Cython extension
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "core.hll_core",
        sources=["core/hll_core.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]

setup(
    name="fractal_manifold",
    ext_modules=cythonize(extensions, language_level=3),
)
