#! /usr/bin/env python
#
def burgers_viscous_time_exact1 ( nu, vxn, vx, vtn, vt ):

#*****************************************************************************80
#
## BURGERS_VISCOUS_TIME_EXACT1 evaluates a solution to the Burgers equation.
#
#  Discussion:
#
#    The form of the Burgers equation considered here is
#
#      du       du        d^2 u
#      -- + u * -- = nu * -----
#      dt       dx        dx^2
#
#    for -1.0 < x < +1.0, and 0 < t.
#
#    Initial conditions are u(x,0) = - sin(pi*x).  Boundary conditions
#    are u(-1,t) = u(+1,t) = 0.  The viscosity parameter nu is taken
#    to be 0.01 / pi, although this is not essential.
#
#    The authors note an integral representation for the solution u(x,t),
#    and present a better version of the formula that is amenable to
#    approximation using Hermite quadrature.
#
#    This program library does little more than evaluate the exact solution
#    at a user-specified set of points, using the quadrature rule.
#    Internally, the order of this quadrature rule is set to 8, but the
#    user can easily modify this value if greater accuracy is desired.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    24 September 2015
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
#    Input, integer VXN, the number of spatial grid points.
#
#    Input, real VX(VXN), the spatial grid points.
#
#    Input, integer VTN, the number of time grid points.
#
#    Input, real VT(VTN), the time grid points.
#
#    Output, real VU(VXN,VTN), the solution of the Burgers
#    equation at each space and time grid point.
#
  import numpy as np
  from hermite_ek_compute import hermite_ek_compute

  qn = 8
#
#  Compute the rule.
#
  qx, qw = hermite_ek_compute ( qn )
#
#  Evaluate U(X,T) for later times.
#
  vu = np.zeros ( [ vxn, vtn ] )

  for vti in range ( 0, vtn ):

    if ( vt[vti] == 0.0 ):

      for i in range ( 0, vxn ):
        vu[i,vti] = - np.sin ( np.pi * vx[i] )

    else:

      for vxi in range ( 0, vxn ):

        top = 0.0
        bot = 0.0

        for qi in range ( 0, qn ):

          c = 2.0 * np.sqrt ( nu * vt[vti] )

          top = top - qw[qi] * c * np.sin ( np.pi * ( vx[vxi] - c * qx[qi] ) ) \
            * np.exp ( - np.cos ( np.pi * ( vx[vxi] - c * qx[qi]  ) ) \
            / ( 2.0 * np.pi * nu ) )

          bot = bot + qw[qi] * c \
            * np.exp ( - np.cos ( np.pi * ( vx[vxi] - c * qx[qi]  ) ) \
            / ( 2.0 * np.pi * nu ) )

          vu[vxi,vti] = top / bot

  return vu

def burgers_viscous_time_exact1_test01 ( ):

#*****************************************************************************80
#
##  BURGERS_VISCOUS_TIME_EXACT1_TEST01 tests sets up a small test case.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    24 September 2015
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
  nu = 0.01 / np.pi

  print ( '' )
  print ( 'BURGERS_VISCOUS_TIME_EXACT1_TEST01' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  BURGERS_VISCOUS_TIME_EXACT1 evaluates solution #1' )
  print ( '  to the Burgers equation.' )
  print ( '' )
  print ( '  Viscosity NU = %g' % ( nu ) )
  print ( '  NX = %d' % ( vxn ) )
  print ( '  NT = %d' % ( vtn ) )

  xlo = -1.0
  xhi = +1.0
  vx = np.linspace ( xlo, xhi, vxn )
  r8vec_print ( vxn, vx, '  X grid points:' )

  tlo = 0.0
  thi = 3.0 / np.pi
  vt = np.linspace ( tlo, thi, vtn )
  r8vec_print ( vtn, vt, '  T grid points:' )

  vu = burgers_viscous_time_exact1 ( nu, vxn, vx, vtn, vt )

  r8mat_print ( vxn, vtn, vu, '  U(X,T) at grid points:' )

  filename = 'burgers_solution_test01.txt'

  r8mat_write ( filename, vxn, vtn, vu )

  print ( '' )
  print ( '  Data written to file "%s"' % ( filename ) )
#
#  Terminate
#
  print ( '' )
  print ( 'BURGERS_VISCOUS_TIME_EXACT1_TEST01' )
  print ( '  Normal end of execution.' )
  return

def burgers_viscous_time_exact1_test02 ( ):

#*****************************************************************************80
#
## BURGERS_VISCOUS_TIME_EXACT1_TEST02 tests sets up a finer test case.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    24 September 2015
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
  nu = 0.01 / np.pi

  print ( '' )
  print ( 'BURGERS_VISCOUS_TIME_EXACT1_TEST02' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  BURGERS_VISCOUS_TIME_EXACT1 computes solution #1' )
  print ( '  to the Burgers equation.' )
  print ( '' )
  print ( '  Viscosity NU = %g' % ( nu ) )
  print ( '  NX = %d' % ( vxn ) )
  print ( '  NT = %d' % ( vtn ) )

  xlo = -1.0
  xhi = +1.0
  vx = np.linspace ( xlo, xhi, vxn )
  r8vec_print ( vxn, vx, '  X grid points:' )

  tlo = 0.0
  thi = 3.0 / np.pi
  vt = np.linspace ( tlo, thi, vtn )
  r8vec_print ( vtn, vt, '  T grid points:' )

  vu = burgers_viscous_time_exact1 ( nu, vxn, vx, vtn, vt )

  filename = 'burgers_solution_test02.txt'

  r8mat_write ( filename, vxn, vtn, vu )

  print ( '' )
  print ( '  Data written to file "%s"' % ( filename ) )
#
#  Terminate
#
  print ( '' )
  print ( 'BURGERS_VISCOUS_TIME_EXACT1_TEST02' )
  print ( '  Normal end of execution.' )
  return

if ( __name__ == '__main__' ):
  from timestamp import timestamp
  timestamp ( )
  burgers_viscous_time_exact1_test01 ( )
  burgers_viscous_time_exact1_test02 ( )
  timestamp ( )
