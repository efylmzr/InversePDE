
import drjit as dr
import mitsuba as mi

import os
PATH = os.path.dirname((os.path.dirname(__file__)))

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

