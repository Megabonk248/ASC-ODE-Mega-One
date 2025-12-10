#include <iostream>
#include <autodiff.hpp>
#include <vector>


using namespace ASC_ode;


template <typename T>
T func1 (T x, T y)
{
  return x * sin(y);
  // return 1e6 + y;
}

template <typename T>
void LegendrePolynomials(int n, T x, std::vector<AutoDiff<1, T>>& P) {
    if (n < 0) {
        P.clear();
        return;
    }
    P.resize(n + 1);
    P[0] = { Variable<0>(T(1)) };
    if (n == 0) return;
    P[1] = { Variable<0>(x) };
    for (int k = 2; k <= n; ++k) {
        P[k] = ((T(2 * k - 1) * x * P[k - 1]) - T(k - 1) * P[k - 2]) / T(k);
    }
}



int main()
{
  double x = 1, y = 2;
  AutoDiff<2> adx = Variable<0>(x);
  AutoDiff<2> ady = Variable<1>(y);

  std::cout << "adx = " << adx << std::endl;
  std::cout << "ady = " << ady << std::endl;

  AutoDiff<2> prod = adx * ady;
  std::cout << "prod = " << prod << std::endl;

  std::cout << "func1(adx, ady) = " << func1(adx, ady) << std::endl;

  double eps = 1e-8;
  std::cout << "numdiff df/dx = " << (func1(x + eps, y) - func1(x-eps, y)) / (2*eps) << std::endl;
  std::cout << "numdiff df/dy = " << (func1(x, y + eps) - func1(x, y-eps)) / (2*eps) << std::endl;


  std::vector<AutoDiff<1, double>> poly(5);
  LegendrePolynomials(5, 2.0, poly);
  for (int i = 0; i < 5; i++) {
    std::cout << poly[i] << std::endl;
  }


  {
    // we can do second derivatives:
    AutoDiff<1, AutoDiff<1>> addx{Variable<0>(2)};
    std::cout << "addx = " << addx << std::endl;
    // func = x*x
    // func' = 2*x
    // func'' = 2
    std::cout << "addx*addx = " << addx * addx << std::endl;

    // std::cout << "sin(addx) = " << sin(addx) << std::endl;
  }
  return 0;
}