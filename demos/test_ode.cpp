#include <iostream>
#include <fstream> 

#include <nonlinfunc.hpp>
#include <timestepper.hpp>

using namespace ASC_ode;


class MassSpring : public NonlinearFunction
{
private:
  double mass;
  double stiffness;

public:
  MassSpring(double m, double k) : mass(m), stiffness(k) {}

  size_t dimX() const override { return 2; }
  size_t dimF() const override { return 2; }
  
  void evaluate (VectorView<double> x, VectorView<double> f) const override
  {
    f(0) = x(1);
    f(1) = -stiffness/mass*x(0);
  }
  
  void evaluateDeriv (VectorView<double> x, MatrixView<double> df) const override
  {
    df = 0.0;
    df(0,1) = 1;
    df(1,0) = -stiffness/mass;
  }
};

/*class KondensatorSpannung : public NonlinearFunction
{
private:
  double widerstand;
  double kapazität;

public:
  KondensatorSpannung(double w, double k) : widerstand(w), kapazität(k) {}

  size_t dimX() const override { return 1; }
  size_t dimF() const override { return 1; }
  
  void evaluate (VectorView<double> x, VectorView<double> f) const override
  {
    f(0) = (U_0(t) - x(0)) / widerstand * kapazität;
  }
  
  void evaluateDeriv (VectorView<double> x, MatrixView<double> df) const override
  {
    df = 0.0;
    df(0,1) = 1;
    df(1,0) = -stiffness/mass;
  }
};*/


int main()
{
  double tend = 4*3.141592*10;
  int steps = 10000;
  double tau = tend/steps;

  Vector<> y = { 1, 0 };  // initializer list
  auto rhs = std::make_shared<MassSpring>(1.0, 1.0);
  
  // ExplicitEuler stepper(rhs);
  // ImplicitEuler stepper(rhs);
  Crank stepper(rhs);

  std::ofstream outfile ("output_test_ode.txt");
  std::cout << 0.0 << "  " << y(0) << " " << y(1) << std::endl;
  outfile << 0.0 << "  " << y(0) << " " << y(1) << std::endl;

  for (int i = 0; i < steps; i++)
  {
     stepper.doStep(tau, y);

     std::cout << (i+1) * tau << "  " << y(0) << " " << y(1) << std::endl;
     outfile << (i+1) * tau << "  " << y(0) << " " << y(1) << std::endl;
  }
}
