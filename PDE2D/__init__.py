
import drjit as dr
import mitsuba as mi
from enum import IntEnum


import os
PATH = os.path.dirname((os.path.dirname(__file__)))

#from  drjit.cuda.ad import (ArrayXu, ArrayXu64, ArrayXf, ArrayXf64, ArrayXb, Array4u64)

#double_precision = True
#source = dr.cuda.ad

#vars = ["ArrayXu", "ArrayXu64", "ArrayXf", "ArrayXf64", "ArrayXb", "Array4u64"]

#if double_precision:
#    vars_precision = ["ArrayXu", "ArrayXu64", "ArrayXf64", "ArrayXf64", "ArrayXb", "Array4u64"]
#else:
#    vars_precision = ["ArrayXu64", "ArrayXu64", "ArrayXf", "ArrayXf64", "ArrayXb", "Array4u64"]

#for name, name_precision in zip(vars, vars_precision):
#    globals()[name] = getattr(source, name_precision)
ArrayXf = None
ArrayXu = None
if "double" in mi.variant():
    ArrayXf = dr.cuda.ad.ArrayXf64 if "cuda" in mi.variant() else dr.llvm.ad.ArrayXf64
    ArrayXu = dr.cuda.ad.ArrayXu64 if "cuda" in mi.variant() else dr.llvm.ad.ArrayXu64
else:
    ArrayXf = dr.cuda.ad.ArrayXf if "cuda" in mi.variant() else dr.llvm.ad.ArrayXf
    ArrayXu = dr.cuda.ad.ArrayXu if "cuda" in mi.variant() else dr.llvm.ad.ArrayXu

ArrayXu64 =  dr.cuda.ad.ArrayXu64 if "cuda" in mi.variant() else dr.llvm.ad.ArrayXu64
ArrayXf64 =  dr.cuda.ad.ArrayXf64 if "cuda" in mi.variant() else dr.llvm.ad.ArrayXf64
Array4u64 =  dr.cuda.ad.Array4u64 if "cuda" in mi.variant() else dr.llvm.ad.Array4u64
ArrayXb =  dr.cuda.ad.ArrayXb if "cuda" in mi.variant() else dr.llvm.ad.ArrayXb

class DIM(IntEnum):
    Two = 0,
    Three = 1

class GreenSampling(IntEnum):
    Polynomial = 0,
    Analytic = 1

class Split(IntEnum):
    Naive = 0,
    Normal = 1,
    Agressive = 2