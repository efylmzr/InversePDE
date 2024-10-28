import drjit as dr
import mitsuba as mi
import numpy as np
from math import nan,inf
from mitsuba import Float

use_special_form = True
#C = 5
A_i0 = Float([
    -4.41534164647933937950E-18, 3.33079451882223809783E-17,
    -2.43127984654795469359E-16, 1.71539128555513303061E-15,
    -1.16853328779934516808E-14, 7.67618549860493561688E-14,
    -4.85644678311192946090E-13, 2.95505266312963983461E-12,
    -1.72682629144155570723E-11, 9.67580903537323691224E-11,
    -5.18979560163526290666E-10, 2.65982372468238665035E-9,
    -1.30002500998624804212E-8, 6.04699502254191894932E-8,
    -2.67079385394061173391E-7, 1.11738753912010371815E-6,
    -4.41673835845875056359E-6, 1.64484480707288970893E-5,
    -5.75419501008210370398E-5, 1.88502885095841655729E-4,
    -5.76375574538582365885E-4, 1.63947561694133579842E-3,
    -4.32430999505057594430E-3, 1.05464603945949983183E-2,
    -2.37374148058994688156E-2, 4.93052842396707084878E-2,
    -9.49010970480476444210E-2, 1.71620901522208775349E-1,
    -3.04682672343198398683E-1, 6.76795274409476084995E-1])

B_i0 = Float([
    -7.23318048787475395456E-18, -4.83050448594418207126E-18,
    4.46562142029675999901E-17, 3.46122286769746109310E-17,
    -2.82762398051658348494E-16, -3.42548561967721913462E-16,
    1.77256013305652638360E-15, 3.81168066935262242075E-15,
    -9.55484669882830764870E-15, -4.15056934728722208663E-14,
    1.54008621752140982691E-14, 3.85277838274214270114E-13,
    7.18012445138366623367E-13, -1.79417853150680611778E-12,
    -1.32158118404477131188E-11, -3.14991652796324136454E-11,
    1.18891471078464383424E-11, 4.94060238822496958910E-10,
    3.39623202570838634515E-9, 2.26666899049817806459E-8,
    2.04891858946906374183E-7, 2.89137052083475648297E-6,
    6.88975834691682398426E-5, 3.36911647825569408990E-3, 
    8.04490411014108831608E-1])

A_k0 = Float([
    1.37446543561352307156E-16, 4.25981614279661018399E-14,
    1.03496952576338420167E-11, 1.90451637722020886025E-9,
    2.53479107902614945675E-7, 2.28621210311945178607E-5,
    1.26461541144692592338E-3, 3.59799365153615016266E-2,
    3.44289899924628486886E-1, -5.35327393233902768720E-1])

B_k0 = Float([
    5.30043377268626276149E-18, -1.64758043015242134646E-17,
    5.21039150503902756861E-17, -1.67823109680541210385E-16,
    5.51205597852431940784E-16, -1.84859337734377901440E-15,
    6.34007647740507060557E-15, -2.22751332699166985548E-14,
    8.03289077536357521100E-14, -2.98009692317273043925E-13,
    1.14034058820847496303E-12, -4.51459788337394416547E-12,
    1.85594911495471785253E-11, -7.95748924447710747776E-11,
    3.57739728140030116597E-10, -1.69753450938905987466E-9,
    8.57403401741422608519E-9,  -4.66048989768794782956E-8,
    2.76681363944501510342E-7,  -1.83175552271911948767E-6,
    1.39498137188764993662E-5,  -1.28495495816278026384E-4,
    1.56988388573005337491E-3,  -3.14481013119645005427E-2, 
    2.44030308206595545468E0])

A_i1 = Float([
    2.77791411276104639959E-18, -2.11142121435816608115E-17,
    1.55363195773620046921E-16, -1.10559694773538630805E-15,
    7.60068429473540693410E-15, -5.04218550472791168711E-14,
    3.22379336594557470981E-13, -1.98397439776494371520E-12,
    1.17361862988909016308E-11, -6.66348972350202774223E-11,
    3.62559028155211703701E-10, -1.88724975172282928790E-9,
    9.38153738649577178388E-9, -4.44505912879632808065E-8,
    2.00329475355213526229E-7, -8.56872026469545474066E-7,
    3.47025130813767847674E-6, -1.32731636560394358279E-5,
    4.78156510755005422638E-5, -1.61760815825896745588E-4,
    5.12285956168575772895E-4, -1.51357245063125314899E-3,
    4.15642294431288815669E-3, -1.05640848946261981558E-2,
    2.47264490306265168283E-2, -5.29459812080949914269E-2,
    1.02643658689847095384E-1, -1.76416518357834055153E-1,
    2.52587186443633654823E-1])

B_i1 = Float([
    7.51729631084210481353E-18, 4.41434832307170791151E-18,
    -4.65030536848935832153E-17, -3.20952592199342395980E-17,
    2.96262899764595013876E-16, 3.30820231092092828324E-16
    -1.88035477551078244854E-15, -3.81440307243700780478E-15,
    1.04202769841288027642E-14, 4.27244001671195135429E-14,
    -2.10154184277266431302E-14,-4.08355111109219731823E-13,
    -7.19855177624590851209E-13,2.03562854414708950722E-12,
    1.41258074366137813316E-11, 3.25260358301548823856E-11,
    -1.89749581235054123450E-11, -5.58974346219658380687E-10,
    -3.83538038596423702205E-9, -2.63146884688951950684E-8,
    -2.51223623787020892529E-7,-3.88256480887769039346E-6,
    -1.10588938762623716291E-4, -9.76109749136146840777E-3,
    7.78576235018280120474E-1])

A_k1 = Float([
    -7.02386347938628759343E-18, -2.42744985051936593393E-15,
    -6.66690169419932900609E-13,-1.41148839263352776110E-10,
    -2.21338763073472585583E-8,-2.43340614156596823496E-6,
    -1.73028895751305206302E-4,-6.97572385963986435018E-3,
    -1.22611180822657148235E-1,-3.53155960776544875667E-1,
    1.52530022733894777053E0])

B_k1 = Float([
    -5.75674448366501715755E-18,1.79405087314755922667E-17,
    -5.68946255844285935196E-17,1.83809354436663880070E-16,
    -6.05704724837331885336E-16,2.03870316562433424052E-15,
    -7.01983709041831346144E-15,2.47715442448130437068E-14,
    -8.97670518232499435011E-14,3.34841966607842919884E-13,
    -1.28917396095102890680E-12,5.13963967348173025100E-12,
    -2.12996783842756842877E-11,9.21831518760500529508E-11,
    -4.19035475934189648750E-10,2.01504975519703286596E-9,
    -1.03457624656780970260E-8,5.74108412545004946722E-8,
    -3.50196060308781257119E-7,2.40648494783721712015E-6,
    -1.93619797416608296024E-5,1.95215518471351631108E-4,
    -2.85781685962277938680E-3,1.03923736576817238437E-1,
    2.72062619048444266945E0])

""" Implementation of some special functions."""

""" Evaluate Chebyshev polynomial at x/2 argument."""
@dr.syntax
def chebyshev(x , coeffs):
    b0 = Float(coeffs[0])
    b1 = Float(0) 
    b2 = Float(0) 
    i = 0
    while dr.hint(i < dr.width(coeffs), mode = 'scalar'):
        b2 = b1
        b1 = b0
        b0 = x * b1 - (b2 - coeffs[i])
        i += 1
    return (b0 - b2) * 0.5;

@dr.syntax
def Gσ(y : Float, z : Float) -> Float: # Multiplied with 2*pi version
    result = Float(0)
    #if dr.hint(use_special_form, mode = 'scalar'):
    c_i0_yz_below8 = chebyshev(y * z/2.0 - 2.0, A_i0)
    c_i0_z_above8 = chebyshev(32.0/z - 2.0, B_i0)
    i0_div = Float(0)
    c_i0_yz_above8 = Float(0)
    c_i0_z_below8 = Float(0)
    if (y * z > 8):
        c_i0_yz_above8 = chebyshev(32.0/(y * z) - 2.0, B_i0)
        i0_div = c_i0_yz_above8/c_i0_z_above8 / dr.sqrt(y)
    elif (z <= 8):
        c_i0_z_below8 = chebyshev(z/2.0 - 2.0, A_i0)
        i0_div =  c_i0_yz_below8/c_i0_z_below8 
    else:
        i0_div = c_i0_yz_below8/c_i0_z_above8 / dr.sqrt(z)
    result = (k0(y * z)  -  k0(z) * i0_div * dr.exp((y-1) * z))
    #else:
    #    result = k0(y * z) - k0(z) * i0(y * z) / i0(z)

    valid_region = (y<1) & (y>0)
    result = dr.select(valid_region, result, 0)
    return result

@dr.syntax
def dGσ(y: Float, z: Float) -> Float: # Multiplied with 2*pi version
    result = Float(0)
    #if use_special_form:
    c_i1_yz_below8 = chebyshev(y * z/2.0 - 2.0, A_i1)
    c_i1_z_above8 = chebyshev(32.0/z - 2.0, B_i1)
    i1_div = Float(0)
    c_i1_yz_above8 = Float(0)
    c_i1_z_below8 = Float(0)
    if (y * z > 8):
        c_i1_yz_above8 = chebyshev(32.0/(y * z) - 2.0, B_i1)
        i1_div = c_i1_yz_above8/c_i1_z_above8  / dr.sqrt(y)
    elif z <= 8:
        c_i1_z_below8 = chebyshev(z/2.0 - 2.0, A_i1)
        i1_div = c_i1_yz_below8/c_i1_z_below8 * y
    else:
        i1_div = c_i1_yz_below8/c_i1_z_above8 * y * z * dr.sqrt(z)
    
    result = k1(y * z) -  k1(z) * i1_div * dr.exp((y-1) * z)
    #else:
    #    result = k1(y * z)  -  k1(z) * i1(y * z) / i1(z)

    valid_region = (y<1) & (y>=0)
    result = dr.select(valid_region, result, 0)
    return result

@dr.syntax
def Gσr_int(y : Float, z : Float) -> Float: # This returns sigma |G|.
    #if use_special_form:
    c_i1_yz_below8 = chebyshev(y * z/2.0 - 2.0, A_i1)
    c_i0_z_above8 = chebyshev(32.0/z - 2.0, B_i0)
    c_i1_yz_above8 = Float(0)
    c_i0_z_below8 = Float(0)
    i_div = Float(0)
    if y * z > 8:
        c_i1_yz_above8 = chebyshev(32.0/(y * z) - 2.0, B_i1)
        i_div = c_i1_yz_above8/c_i0_z_above8 / dr.sqrt(y)
    elif z <= 8:
        c_i0_z_below8 = chebyshev(z/2.0 - 2.0, A_i0)
        i_div = c_i1_yz_below8/c_i0_z_below8 * z * y
    else:
        i_div = c_i1_yz_below8/c_i0_z_above8 * z * y * dr.sqrt(z)
    result = 1 - y * z * (i_div * k0(z) * dr.exp((y-1) * z) + k1(y * z))
    #else:
    #    result = 1 - y * z * (k0(z) * i1(y * z) / i0(z) + k1(y * z))
    return result

def Qσ(y,z):
    return 1 - Gσr_int(y,z) * dr.square(z)

@dr.syntax
def i0(x_ : Float) -> Float:
    x = dr.abs(x_)
    x = dr.select(x == 0, dr.epsilon(Float), x)
    if x<=8:
        result = chebyshev(x/2.0 - 2.0, A_i0)
    else:
        result = chebyshev(32.0/x - 2.0, B_i0) / dr.sqrt(x)
    
    
    if x_==0:
        result =  Float(1)
    else:
        result *= dr.exp(x)
    return result

@dr.syntax
def k0(x_):
    x = dr.abs(x_)
    
    if x == Float(0):
        x = Float(dr.epsilon(Float))
    
    if x <= 2:
        result = chebyshev((dr.sqr(x) - 2), A_k0) - dr.log(0.5 * x) * i0(x)
    else:
        result = dr.exp(-x) * chebyshev(8.0 / x - 2, B_k0) / dr.sqrt(x)
    if x_ == 0:
        result = Float(dr.inf)
    if x_ < 0:
        result = Float(dr.nan)
    return result

@dr.syntax
def i1(x_):
    x = dr.abs(x_)
    if x == 0:
        x += dr.epsilon(Float)

    if x<=8:
        result = chebyshev(x/2.0 - 2.0, A_i1) * x
    else:
        result = chebyshev(32.0 / x - 2.0, B_i1) / dr.sqrt(x)
    
    if x_ == 0:
        result = Float(0)
    if x_ < 0:
        result = -result
    return result  * dr.exp(x)

@dr.syntax
def k1(x_):
    x = dr.abs(x_)
    
    if x == 0:
        x += dr.epsilon(Float)

    if x <= 2:
        result = dr.log(x * 0.5) * i1(x) + chebyshev(dr.sqr(x) - 2.0, A_k1) / x
    else:
        result = dr.exp(-x) * chebyshev(8.0 / x -2, B_k1) / dr.sqrt(x)
    if x_ == 0:
        result = Float(dr.inf)
    if x_ < 0:
        result = Float(dr.nan)
    return result


## Remove the exponential terms

# divided by exp(x)
@dr.syntax
def i0_(x_):
    x = dr.abs(x_)
    x = dr.select(x == 0, dr.epsilon(Float), x)
    if x<=8:
        result = chebyshev(x/2.0 - 2.0, A_i0)
    else:
        result = chebyshev(32.0/x - 2.0, B_i0) / dr.sqrt(x)
    
    if x_==0:
        result = Float(1)
    return result

# multiplied by exp(x)
@dr.syntax
def k0_(x_):
    x = dr.abs(x_)
    if x == 0:
        x += dr.epsilon(Float)
    if x <= 2:
        result = dr.exp(x) (chebyshev((dr.sqr(x) - 2), A_k0) - dr.log(0.5 * x) * i0(x))
    else:
        result =  chebyshev(8.0 / x - 2, B_k0) / dr.sqrt(x),
    if x_ == 0:
        result = Float(dr.inf)
    if x_ < 0:
        result = Float(dr.nan)
    return result

# divided by exp(x)
@dr.syntax
def i1_(x_):
    x = dr.abs(x_)
    if x == 0:
        x += dr.epsilon(Float)

    if x<=8:
        result = chebyshev(x/2.0 - 2.0, A_i1) * x
    else:
        result = chebyshev(32.0 / x - 2.0, B_i1) / dr.sqrt(x)
    
    if x_ == 0:
        result = Float(0)
    if x_ < 0:
        result = -result
    return result 

# multiplied by exp(x)
@dr.syntax
def k1_(x_):
    x = dr.abs(x_)
    
    if x == 0:
        x += dr.epsilon(Float)

    if x <= 2:
        result = dr.exp(x) * (dr.log(x * 0.5) * i1(x) + chebyshev(dr.sqr(x) - 2.0, A_k1) / x)
    else:
        result =  chebyshev(8.0 / x -2, B_k1) / dr.sqrt(x)
    if x_ == 0:
        result = Float(inf)
    if x_ < 0:
        result = Float(nan)
    return result


    
