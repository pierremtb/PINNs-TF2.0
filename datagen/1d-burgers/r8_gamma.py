#! /usr/bin/env python
#
def r8_gamma ( x ):

#*****************************************************************************80
#
## R8_GAMMA evaluates Gamma(X) for a real argument.
#
#  Discussion:
#
#    This routine calculates the gamma function for a real argument X.
#
#    Computation is based on an algorithm outlined in reference 1.
#    The program uses rational functions that approximate the gamma
#    function to at least 20 significant decimal digits.  Coefficients
#    for the approximation over the interval (1,2) are unpublished.
#    Those for the approximation for 12 <= X are from reference 2.
#
#    PYTHON provides a GAMMA function, which is likely to be faster, and more
#    accurate.  
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    24 July 2014
#
#  Author:
#
#    Original FORTRAN77 version by William Cody, Laura Stoltz.
#    PYTHON version by John Burkardt.
#
#  Reference:
#
#    William Cody,
#    An Overview of Software Development for Special Functions,
#    in Numerical Analysis Dundee, 1975,
#    edited by GA Watson,
#    Lecture Notes in Mathematics 506,
#    Springer, 1976.
#
#    John Hart, Ward Cheney, Charles Lawson, Hans Maehly,
#    Charles Mesztenyi, John Rice, Henry Thatcher,
#    Christoph Witzgall,
#    Computer Approximations,
#    Wiley, 1968,
#    LC: QA297.C64.
#
#  Parameters:
#
#    Input, real X, the argument of the function.
#
#    Output, real VALUE, the value of the function.
#
  import numpy as np
  from math import exp
  from math import floor
  from math import log
  from math import sin
#
#  Coefficients for minimax approximation over (12, INF).
#
  c = np.array ( [
   -1.910444077728E-03, \
    8.4171387781295E-04, \
   -5.952379913043012E-04, \
    7.93650793500350248E-04, \
   -2.777777777777681622553E-03, \
    8.333333333333333331554247E-02, \
    5.7083835261E-03 ] )
#
#  Mathematical constants
#
  r8_pi = 3.141592653589793
  sqrtpi = 0.9189385332046727417803297
#
#  Machine dependent parameters
#
  xbig = 171.624
  xminin = 2.23E-308
  eps = 2.22E-16
  xinf = 1.79E+308
#
#  Numerator and denominator coefficients for rational minimax
#  approximation over (1,2).
#
  p = np.array ( [ \
   -1.71618513886549492533811E+00, \
    2.47656508055759199108314E+01, \
   -3.79804256470945635097577E+02, \
    6.29331155312818442661052E+02, \
    8.66966202790413211295064E+02, \
   -3.14512729688483675254357E+04, \
   -3.61444134186911729807069E+04, \
    6.64561438202405440627855E+04 ] )

  q = np.array ( [ \
   -3.08402300119738975254353E+01, \
    3.15350626979604161529144E+02, \
   -1.01515636749021914166146E+03, \
   -3.10777167157231109440444E+03, \
    2.25381184209801510330112E+04, \
    4.75584627752788110767815E+03, \
   -1.34659959864969306392456E+05, \
   -1.15132259675553483497211E+05 ] )

  parity = 0
  fact = 1.0
  n = 0
  y = x
#
#  Argument is negative.
#
  if ( y <= 0.0 ):

    y = - x
    y1 = floor ( y )
    res = y - y1

    if ( res != 0.0 ):

      if ( y1 != floor ( y1 * 0.5 ) * 2.0 ):
        parity = 1

      fact = - r8_pi / sin ( r8_pi * res )
      y = y + 1.0

    else:

      res = xinf
      value = res
      return value
#
#  Argument is positive.
#
  if ( y < eps ):
#
#  Argument < EPS.
#
    if ( xminin <= y ):
      res = 1.0 / y
    else:
      res = xinf

    value = res
    return value

  elif ( y < 12.0 ):

    y1 = y
#
#  0.0 < argument < 1.0.
#
    if ( y < 1.0 ):

      z = y
      y = y + 1.0
#
#  1.0 < argument < 12.0.
#  Reduce argument if necessary.
#
    else:

      n = int ( floor ( y ) - 1 )
      y = y - n
      z = y - 1.0
#
#  Evaluate approximation for 1.0 < argument < 2.0.
#
    xnum = 0.0
    xden = 1.0
    for i in range ( 0, 8 ):
      xnum = ( xnum + p[i] ) * z
      xden = xden * z + q[i]

    res = xnum / xden + 1.0
#
#  Adjust result for case  0.0 < argument < 1.0.
#
    if ( y1 < y ):

      res = res / y1
#
#  Adjust result for case 2.0 < argument < 12.0.
#
    elif ( y < y1 ):

      for i in range ( 0, n ):
        res = res * y
        y = y + 1.0

  else:
#
#  Evaluate for 12.0 <= argument.
#
    if ( y <= xbig ):

      ysq = y * y
      sum = c[6]
      for i in range ( 0, 6 ):
        sum = sum / ysq + c[i]
      sum = sum / y - y + sqrtpi
      sum = sum + ( y - 0.5 ) * log ( y )
      res = exp ( sum )

    else:

      res = xinf
      value = res
      return value
#
#  Final adjustments and return.
#
  if ( parity ):
    res = - res

  if ( fact != 1.0 ):
    res = fact / res

  value = res

  return value

def r8_gamma_test ( ):

#*****************************************************************************80
#
## R8_GAMMA_TEST demonstrates the use of R8_GAMMA.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    25 July 2014
#
#  Author:
#
#    John Burkardt
#
  import platform
  from gamma_values import gamma_values

  print ( '' )
  print ( 'R8_GAMMA_TEST:' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  R8_GAMMA evaluates the Gamma function.' )
  print ( '' )
  print ( '      X            GAMMA(X)      R8_GAMMA(X)' )
  print ( '' )

  n_data = 0

  while ( True ):

    n_data, x, fx1 = gamma_values ( n_data )

    if ( n_data == 0 ):
      break

    fx2 = r8_gamma ( x )

    print ( '  %12g  %24.16g  %24.16g' % ( x, fx1, fx2 ) )
#
#  Terminate.
#
  print ( '' )
  print ( 'R8_GAMMA_TEST' )
  print ( '  Normal end of execution.' )
  return

if ( __name__ == '__main__' ):
  from timestamp import timestamp
  timestamp ( ) 
  r8_gamma_test ( )
  timestamp ( )

