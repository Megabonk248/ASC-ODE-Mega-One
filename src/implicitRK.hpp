#ifndef IMPLICITRK_HPP
#define IMPLICITRK_HPP

#include <vector.hpp>
#include <matrix.hpp>
#include <inverse.hpp>

namespace ASC_ode {
  using namespace nanoblas;



  class ImplicitRungeKutta : public TimeStepper
  {
    Matrix<> m_a;
    Vector<> m_b, m_c;
    std::shared_ptr<NonlinearFunction> m_equ;
    std::shared_ptr<Parameter> m_tau;
    std::shared_ptr<ConstantFunction> m_yold;
    int m_stages;
    int m_n;
    Vector<> m_k, m_y;
  public:
    ImplicitRungeKutta(std::shared_ptr<NonlinearFunction> rhs,
      const Matrix<> &a, const Vector<> &b, const Vector<> &c) 
    : TimeStepper(rhs), m_a(a), m_b(b), m_c(c),
    m_tau(std::make_shared<Parameter>(0.0)),
    m_stages(c.size()), m_n(rhs->dimX()), m_k(m_stages*m_n), m_y(m_stages*m_n)
    {
      auto multiple_rhs = make_shared<MultipleFunc>(rhs, m_stages);
      m_yold = std::make_shared<ConstantFunction>(m_stages*m_n);
      auto knew = std::make_shared<IdentityFunction>(m_stages*m_n);
      m_equ = knew - Compose(multiple_rhs, m_yold+m_tau*std::make_shared<MatVecFunc>(a, m_n));
    }

    void doStep(double tau, VectorView<double> y) override
    {
      for (int j = 0; j < m_stages; j++)
        m_y.range(j*m_n, (j+1)*m_n) = y;
      m_yold->set(m_y);

      m_tau->set(tau);
      m_k = 0.0;  
      NewtonSolver(m_equ, m_k);

      for (int j = 0; j < m_stages; j++)
        y += tau * m_b(j) * m_k.range(j*m_n, (j+1)*m_n);
    }
  };







Matrix<double> Gauss2a { { 0.25, 0.25 - sqrt(3)/6 }, { 0.25 + sqrt(3)/6, 0.25 } };
Vector<> Gauss2b { 0.5, 0.5 };
Vector<> Gauss2c { 0.5 - sqrt(3)/6, 0.5 + sqrt(3)/6 };


Vector<> Gauss3c { 0.5 - sqrt(15)/10, 0.5, 0.5+sqrt(15)/10 };


// codes from Numerical Recipes, https://numerical.recipes/book.html

// Gauss integration on [0,1]
void GaussLegendre(VectorView<> x, VectorView<> w)
// Given the lower and upper limits of integration x1 and x2, this routine returns arrays x[0..n-1]
// and w[0..n-1] of length n, containing the abscissas and weights of the Gauss-Legendre n-point
// quadrature formula.
  {
    double x1 = 0;
    double x2 = 1;
    const double EPS=1.0e-14;  // EPS is the relative precision.
    double z1,z,xm,xl,pp,p3,p2,p1;
    int n=x.size();
    int m=(n+1)/2;  // The roots are symmetric in the interval, so
    xm=0.5*(x2+x1); // we only have to find half of them.
    xl=0.5*(x2-x1);
    for (int i=0;i<m;i++) {  // Loop over the desired roots.
      z=cos(3.141592654*(i+0.75)/(n+0.5));
      // Starting with this approximation to the ith root, we enter the main loop of refinement
      // by Newton’s method.
      do {
        p1=1.0;
        p2=0.0;
        for (int j=0;j<n;j++) { // Loop up the recurrence relation to get the
            p3=p2;  // Legendre polynomial evaluated at z.
          p2=p1;
          p1=((2.0*j+1.0)*z*p2-j*p3)/(j+1);
        }
        // p1 is now the desired Legendre polynomial. We next compute pp, its derivative,
        // by a standard relation involving also p2, the polynomial of one lower order.
        pp=n*(z*p1-p2)/(z*z-1.0);
        z1=z;
        z=z1-p1/pp;   // Newton’s method.
      } while (abs(z-z1) > EPS);
      x[i]=xm-xl*z;      // Scale the root to the desired interval,
      x[n-1-i]=xm+xl*z;  //  and put in its symmetric counterpart.
      w[i]=2.0*xl/((1.0-z*z)*pp*pp);  // Compute the weight
      w[n-1-i]=w[i];     // and its symmetric counterpart.
    }
  }


void GaussJacobi (VectorView<> x, VectorView<> w, const double alf, const double bet)
// Given alf and bet, the parameters ˛ and ˇ of the Jacobi polynomials, this routine returns
// arrays x[0..n-1] and w[0..n-1] containing the abscissas and weights of the n-point GaussJacobi quadrature formula. The largest abscissa is returned in x[0], the smallest in x[n-1].
{
  const int MAXIT=10;
  const double EPS=1.0e-14; // EPS is the relative precision.
  int i,its,j;
  double alfbet,an,bn,r1,r2,r3;
  double a,b,c,p1,p2,p3,pp,temp,z,z1;
  int n=x.size();
  for (i=0;i<n;i++) { // Loop over the desired roots.
    if (i == 0) {  // Initial guess for the largest root.
      an=alf;
      bn=bet/n;
      r1=(1.0+alf)*(2.78/(4.0+n*n)+0.768*an/n);
      r2=1.0+1.48*an+0.96*bn+0.452*an*an+0.83*an*bn;
      z=1.0-r1/r2;
    } else if (i == 1) { // Initial guess for the second largest root.
      r1=(4.1+alf)/((1.0+alf)*(1.0+0.156*alf));
      r2=1.0+0.06*(n-8.0)*(1.0+0.12*alf)/n;
      r3=1.0+0.012*bet*(1.0+0.25*abs(alf))/n;
      z -= (1.0-z)*r1*r2*r3;
    } else if (i == 2) { // Initial guess for the third largest root.
      r1=(1.67+0.28*alf)/(1.0+0.37*alf);
      r2=1.0+0.22*(n-8.0)/n;
      r3=1.0+8.0*bet/((6.28+bet)*n*n);
      z -= (x[0]-z)*r1*r2*r3;
    } else if (i == n-2) { // Initial guess for the second smallest root.
      r1=(1.0+0.235*bet)/(0.766+0.119*bet);
      r2=1.0/(1.0+0.639*(n-4.0)/(1.0+0.71*(n-4.0)));
      r3=1.0/(1.0+20.0*alf/((7.5+alf)*n*n));
      z += (z-x[n-4])*r1*r2*r3;
    } else if (i == n-1) { // Initial guess for the smallest root.
      r1=(1.0+0.37*bet)/(1.67+0.28*bet);
      r2=1.0/(1.0+0.22*(n-8.0)/n);
      r3=1.0/(1.0+8.0*alf/((6.28+alf)*n*n));
      z += (z-x[n-3])*r1*r2*r3;
    } else { // Initial guess for the other roots.
      z=3.0*x[i-1]-3.0*x[i-2]+x[i-3];
    }
    alfbet=alf+bet;
    for (its=1;its<=MAXIT;its++) { // Refinement by Newton’s method.
      temp=2.0+alfbet; // Start the recurrence with P0 and P1 to avoid
      // a division by zero when ˛ C ˇ D 0 or 1.
      p1=(alf-bet+temp*z)/2.0;
      p2=1.0;
      for (j=2;j<=n;j++) { // Loop up the recurrence relation to get the
        p3=p2;    // Jacobi polynomial evaluated at z.
        p2=p1;
        temp=2*j+alfbet;
        a=2*j*(j+alfbet)*(temp-2.0);
        b=(temp-1.0)*(alf*alf-bet*bet+temp*(temp-2.0)*z);
        c=2.0*(j-1+alf)*(j-1+bet)*temp;
        p1=(b*p2-c*p3)/a;
      }
      pp=(n*(alf-bet-temp*z)*p1+2.0*(n+alf)*(n+bet)*p2)/(temp*(1.0-z*z));
      // p1 is now the desired Jacobi polynomial. We next compute pp, its derivative, by
      //  a standard relation involving also p2, the polynomial of one lower order.
      z1=z;
      z=z1-p1/pp; // Newton’s formula.
      if (abs(z-z1) <= EPS) break;
    }
    if (its > MAXIT) throw("too many iterations in gaujac");
    x[i]=z;    // Store the root and the weight.
    w[i]=exp(std::lgamma(alf+n)+std::lgamma(bet+n)-std::lgamma(n+1.0)-
             std::lgamma(n+alfbet+1.0))*temp*pow(2.0,alfbet)/(pp*p2);
  }
}



/*
  given Runge-Kutta nodes c, compute the coefficients a and b
*/
auto computeABfromC (const Vector<> & c)
{
  int s = c.size();
  Matrix<> M(s, s);
  Vector<> tmp(s);
  
  for (int i = 0; i < s; i++)
    for (int j = 0; j < s; j++)
      M(i,j) = std::pow(c(j), i);

  calcInverse(M);
  // M = LapackLU(M).inverse();
  
  for (int i = 0; i < s; i++)
    tmp(i) = 1.0 / (i+1);

  Vector<> b = M * tmp;
  Matrix a(s,s);

  for (int j = 0; j < s; j++)
    {
      for (int i = 0; i < s; i++)
        tmp(i) = std::pow(c(j),i+1) / (i+1);
      a.row(j) = M * tmp;
    }
  /*
  std::cout << "b = " << b << std::endl;
  std::cout << "a = " << a << std::endl;
  */
  return std::tuple { a, b };
}
  

void GaussRadau (VectorView<> x, VectorView<> w)
{
  GaussJacobi (x.range(0, x.size()-1),
               w.range(0, w.size()-1), 1, 0);
  for (int i = 0; i < x.size()-1; i++)
    {
      x(i) = 0.5*(x(i)+1);
      w(i) *= 0.5;
    }
  x(x.size()-1) = 1.0;
  double sum = 0;
  for (int i = 0; i < x.size()-1; i++)
    sum += w(i);
  w(x.size()-1) = 1-sum;
}
}

#endif // IMPLICITRK_HPP