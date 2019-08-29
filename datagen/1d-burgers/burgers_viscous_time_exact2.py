#! /usr/bin/env python
#
def burgers_viscous_time_exact2 ( nu, xn, x, tn, t ):

#*****************************************************************************80
#
## BURGERS_VISCOUS_TIME_EXACT2 evaluates a solution to the Burgers equation.
#
#  Discussion:
#
#    The form of the Burgers equation considered here is
#
#      du       du        d^2 u
#      -- + u * -- = nu * -----
#      dt       dx        dx^2
#
#    for 0.0 < x < 2 Pi, and 0 < t.
#
#    The initial condition is
#
#      u(x,0) = 4 - 2 * nu * dphi(x,0)/dx / phi(x,0)
#
#    where
#
#      phi(x,t) = exp ( - ( x-4*t      ) / ( 4*nu*(t+1) ) )
#               + exp ( - ( x-4*t-2*pi ) / ( 4*nu*(t+1) ) )
#
#    The boundary conditions are periodic:
#
#      u(0,t) = u(2 Pi,t)
#
#    The viscosity parameter nu may be taken to be 0.01, but other values
#    may be chosen.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    27 September 2015
#
#  Author:
#
#    John Burkardt.
#
#  Reference:
#
#    Claude Basdevant, Michel Deville, Pierre Haldenwang, J Lacroix,
#    J Ouazzani, Roger Peyret, Paolo Orlandi, Anthony Patera,
#    Spectral and finite difference solutions of the Burgers equation,
#    Computers and Fluids,
#    Volume 14, Number 1, 1986, pages 23-41.
#
#  Parameters:
#
#    Input, real NU, the viscosity.
#
#    Input, integer XN, the number of spatial grid points.
#
#    Input, real X(XN), the spatial grid points.
#
#    Input, integer TN, the number of time grid points.
#
#    Input, real T(TN), the time grid points.
#
#    Output, real U(XN,TN), the solution of the Burgers
#    equation at each space and time grid point.
#
  import numpy as np

  u = np.zeros ( [ xn, tn ] )

  for j in range ( 0, tn ):

    for i in range ( 0, xn ):

      a = ( x[i] - 4.0 * t[j] )
      b = ( x[i] - 4.0 * t[j] - 2.0 * np.pi )
      c = 4.0 * nu * ( t[j] + 1.0 )
      phi = np.exp ( - a * a / c ) + np.exp ( - b * b / c )
      dphi = - 2.0 * a * np.exp ( - a * a / c ) / c \
             - 2.0 * b * np.exp ( - b * b / c ) / c
      u[i,j] = 4.0 - 2.0 * nu * dphi / phi

  return u

def burgers_viscous_time_exact2_test01 ( ):

#*****************************************************************************80
#
##  BURGERS_VISCOUS_TIME_EXACT2_TEST01 tests sets up a small test case.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    27 September 2015
#
#  Author:
#
#    John Burkardt
#
  import numpy as np
  import platform
  from r8mat_print import r8mat_print
  from r8mat_write import r8mat_write
  from r8vec_print import r8vec_print

  vtn = 11
  vxn = 11
  nu = 0.5

  print ( '' )
  print ( 'BURGERS_VISCOUS_TIME_EXACT2_TEST01' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  BURGERS_VISCOUS_TIME_EXACT2 evaluates solution #2' )
  print ( '  to the Burgers equation.' )
  print ( '' )
  print ( '  Viscosity NU = %g' % ( nu ) )
  print ( '  NX = %d' % ( vxn ) )
  print ( '  NT = %d' % ( vtn ) )

  xlo = 0.0
  xhi = 2.0 * np.pi
  vx = np.linspace ( xlo, xhi, vxn )
  r8vec_print ( vxn, vx, '  X grid points:' )

  tlo = 0.0
  thi = 1.0
  vt = np.linspace ( tlo, thi, vtn )
  r8vec_print ( vtn, vt, '  T grid points:' )

  vu = burgers_viscous_time_exact2 ( nu, vxn, vx, vtn, vt )

  r8mat_print ( vxn, vtn, vu, '  U(X,T) at grid points:' )

  filename = 'burgers_solution_test03.txt'

  r8mat_write ( filename, vxn, vtn, vu )

  print ( '' )
  print ( '  Data written to file "%s"' % ( filename ) )
#
#  Terminate
#
  print ( '' )
  print ( 'BURGERS_VISCOUS_TIME_EXACT2_TEST01' )
  print ( '  Normal end of execution.' )
  return

def burgers_viscous_time_exact2_test02 ( ):

#*****************************************************************************80
#
## BURGERS_VISCOUS_TIME_EXACT2_TEST02 tests sets up a finer test case.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    27 September 2015
#
#  Author:
#
#    John Burkardt
#
  import numpy as np
  import platform
  from r8mat_print import r8mat_print
  from r8mat_write import r8mat_write
  from r8vec_print import r8vec_print

  vtn = 41
  vxn = 41
  nu = 0.5

  print ( '' )
  print ( 'BURGERS_VISCOUS_TIME_EXACT2_TEST02' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  BURGERS_VISCOUS_TIME_EXACT2 computes solution #2' )
  print ( '  to the Burgers equation.' )
  print ( '' )
  print ( '  Viscosity NU = %g' % ( nu ) )
  print ( '  NX = %d' % ( vxn ) )
  print ( '  NT = %d' % ( vtn ) )

  xlo = 0.0
  xhi = 2.0 * np.pi
  vx = np.linspace ( xlo, xhi, vxn )
  r8vec_print ( vxn, vx, '  X grid points:' )

  tlo = 0.0
  thi = 1.0
  vt = np.linspace ( tlo, thi, vtn )
  r8vec_print ( vtn, vt, '  T grid points:' )

  vu = burgers_viscous_time_exact2 ( nu, vxn, vx, vtn, vt )

  filename = 'burgers_solution_test04.txt'

  r8mat_write ( filename, vxn, vtn, vu )

  print ( '' )
  print ( '  Data written to file "%s"' % ( filename ) )
#
#  Terminate
#
  print ( '' )
  print ( 'BURGERS_VISCOUS_TIME_EXACT2_TEST02' )
  print ( '  Normal end of execution.' )
  return

if ( __name__ == '__main__' ):
  from timestamp import timestamp
  timestamp ( )
  burgers_viscous_time_exact2_test01 ( )
  burgers_viscous_time_exact2_test02 ( )
  timestamp ( )
