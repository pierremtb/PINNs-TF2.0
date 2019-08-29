function [ x, w ] = hermite_ek_compute ( n )

%*****************************************************************************80
%
%% HERMITE_EK_COMPUTE computes a Gauss-Hermite quadrature rule.
%
%  Discussion:
%
%    The code uses an algorithm by Elhay and Kautsky.
%
%    The abscissas are the zeros of the N-th order Hermite polynomial.
%
%    The integral:
%
%      integral ( -oo < x < +oo ) exp ( - x * x ) * f(x) dx
%
%    The quadrature rule:
%
%      sum ( 1 <= i <= n ) w(i) * f ( x(i) )
%
%  Licensing:
%
%    This code is distributed under the GNU LGPL license.
%
%  Modified:
%
%    19 April 2011
%
%  Author:
%
%    Original FORTRAN77 version by Sylvan Elhay, Jaroslav Kautsky.
%    MATLAB version by John Burkardt.
%
%  Reference:
%
%    Sylvan Elhay, Jaroslav Kautsky,
%    Algorithm 655: IQPACK, FORTRAN Subroutines for the Weights of
%    Interpolatory Quadrature,
%    ACM Transactions on Mathematical Software,
%    Volume 13, Number 4, December 1987, pages 399-415.
%
%  Parameters:
%
%    Input, integer N, the number of abscissas.
%
%    Output, real X(N), the abscissas.
%
%    Output, real W(N), the weights.
%

%
%  Define the zero-th moment.
%
  zemu = gamma ( 0.5 );
%
%  Define the Jacobi matrix.
%
  bj = zeros ( n, 1 );
  for i = 1 : n
    bj(i) = i / 2.0;
  end
  bj(1:n) = sqrt ( bj(1:n) );

  x = zeros ( n, 1 );

  w = zeros ( n, 1 );
  w(1) = sqrt ( zemu );
%
%  Diagonalize the Jacobi matrix.
%
  [ x, w ] = imtqlx ( n, x, bj, w );

  w(1:n) = w(1:n).^2;

  return
end
