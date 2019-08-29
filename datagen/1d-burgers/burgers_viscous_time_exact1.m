function vu = burgers_viscous_time_exact1 ( nu, vxn, vx, vtn, vt )

%*****************************************************************************80
%
%% BURGERS_VISCOUS_TIME_EXACT1 evaluates a solution to the Burgers equation.
%
%  Discussion:
%
%    The form of the Burgers equation considered here is
%
%      du       du        d^2 u
%      -- + u * -- = nu * -----
%      dt       dx        dx^2
%
%    for -1.0 < x < +1.0, and 0 < t.
%
%    Initial conditions are u(x,0) = - sin(pi*x).  Boundary conditions
%    are u(-1,t) = u(+1,t) = 0.  The viscosity parameter nu is taken
%    to be 0.01 / pi, although this is not essential.
%
%    The authors note an integral representation for the solution u(x,t),
%    and present a better version of the formula that is amenable to
%    approximation using Hermite quadrature.
%
%    This program library does little more than evaluate the exact solution
%    at a user-specified set of points, using the quadrature rule.
%    Internally, the order of this quadrature rule is set to 8, but the
%    user can easily modify this value if greater accuracy is desired.
%
%  Licensing:
%
%    This code is distributed under the GNU LGPL license.
%
%  Modified:
%
%    17 November 2011
%
%  Author:
%
%    John Burkardt.
%
%  Reference:
%
%    Claude Basdevant, Michel Deville, Pierre Haldenwang, J Lacroix,
%    J Ouazzani, Roger Peyret, Paolo Orlandi, Anthony Patera,
%    Spectral and finite difference solutions of the Burgers equation,
%    Computers and Fluids,
%    Volume 14, Number 1, 1986, pages 23-41.
%
%  Parameters:
%
%    Input, real NU, the viscosity.
%
%    Input, integer VXN, the number of spatial grid points.
%
%    Input, real VX(VXN), the spatial grid points.
%
%    Input, integer VTN, the number of time grid points.
%
%    Input, real VT(VTN), the time grid points.
%
%    Output, real VU(VXN,VTN), the solution of the Burgers
%    equation at each space and time grid point.
%
  qn = 8;
%
%  Compute the rule.
%
  [ qx, qw ] = hermite_ek_compute ( qn );
%
%  Evaluate U(X,T) for later times.
%
  vu = zeros ( vxn, vtn );

  for vti = 1 : vtn

    if ( vt(vti) == 0.0 )

      vu(1:vxn,vti) = - sin ( pi * vx(1:vxn) );

    else

      for vxi = 1 : vxn

        top = 0.0;
        bot = 0.0;

        for qi = 1 : qn

          c = 2.0 * sqrt ( nu * vt(vti) );

          top = top - qw(qi) * c * sin ( pi * ( vx(vxi) - c * qx(qi) ) ) ...
            * exp ( - cos ( pi * ( vx(vxi) - c * qx(qi)  ) ) ...
            / ( 2.0 * pi * nu ) );

          bot = bot + qw(qi) * c ...
            * exp ( - cos ( pi * ( vx(vxi) - c * qx(qi)  ) ) ...
            / ( 2.0 * pi * nu ) );

          vu(vxi,vti) = top / bot;

        end

      end

    end

  end

  return
end
