#include "mass_spring.hpp"
#include "Newmark.hpp"

int main()
{
  MassSpringSystem<2> mss;
  mss.setGravity( {0,-9.81} );
  auto fA = mss.addFix( { { 0.0, 0.0 } } );
  auto mA = mss.addMass( { 1, { 1.0, 0.0 } } );
  mss.addDistanceConstraint ( { 1, { fA, mA } }  );

  auto mB = mss.addMass( { 1, { 2.0, 0.0 } } );
  mss.addSpring ( { 1, 20, { mA, mB } } );

  std::cout << "mss: " << std::endl << mss << std::endl;


  double tend = 10;
  double steps = 1000;

  Vector<> x(mss.get_state_vec_size());
  Vector<> dx(mss.get_state_vec_size());
  Vector<> ddx(mss.get_state_vec_size());

  auto mss_func = std::make_shared<MSS_Function<2>> (mss);
  auto mass = std::make_shared<IdentityFunction> (x.size());

  mss.getState (x, dx, ddx);
  
  SolveODE_Alpha(tend, steps, 0.8, x, dx, ddx,  mss_func, mass,
                   [](double t, VectorView<double> x) { std::cout << "t = " << t
                                                             << ", x = " << Vec<5>(x) << std::endl; });
}
