#ifndef TIMERSTEPPER_HPP
#define TIMERSTEPPER_HPP

#include <functional>
#include <exception>

#include "Newton.hpp"


namespace ASC_ode
{
  
  class TimeStepper
  { 
  protected:
    std::shared_ptr<NonlinearFunction> m_rhs;
  public:
    TimeStepper(std::shared_ptr<NonlinearFunction> rhs) : m_rhs(rhs) {}
    virtual ~TimeStepper() = default;
    virtual void doStep(double tau, VectorView<double> y) = 0;
  };

  class ExplicitEuler : public TimeStepper
  {
    Vector<> m_vecf;
  public:
    ExplicitEuler(std::shared_ptr<NonlinearFunction> rhs) 
    : TimeStepper(rhs), m_vecf(rhs->dimF()) {}
    void doStep(double tau, VectorView<double> y) override
    {
      this->m_rhs->evaluate(y, m_vecf);
      y += tau * m_vecf;
    }
  };
  
  class ImprovedEuler : public TimeStepper
  {
    Vector<> m_vecf;
  public:
    ImprovedEuler(std::shared_ptr<NonlinearFunction> rhs) 
    : TimeStepper(rhs), m_vecf(rhs->dimF()) {}
    void doStep(double tau, VectorView<double> y) override
    {
      this->m_rhs->evaluate(y, m_vecf);
      Vector<double> d = (tau / 2.0 * m_vecf) + y;
      this->m_rhs->evaluate(d, m_vecf);
      y += tau * m_vecf;
    }
  };

  class ImplicitEuler : public TimeStepper
  {
    std::shared_ptr<NonlinearFunction> m_equ;
    std::shared_ptr<Parameter> m_tau;
    std::shared_ptr<ConstantFunction> m_yold;
  public:
    ImplicitEuler(std::shared_ptr<NonlinearFunction> rhs) 
    : TimeStepper(rhs), m_tau(std::make_shared<Parameter>(0.0)) 
    {
      m_yold = std::make_shared<ConstantFunction>(rhs->dimX());
      auto ynew = std::make_shared<IdentityFunction>(rhs->dimX());
      m_equ = ynew - m_yold - m_tau * m_rhs;
    }

    void doStep(double tau, VectorView<double> y) override
    {
      m_yold->set(y);
      m_tau->set(tau);
      NewtonSolver(m_equ, y);
    }
  };

  class Crank : public TimeStepper
  {
    std::shared_ptr<NonlinearFunction> m_equ;
    std::shared_ptr<Parameter> m_tau;
    std::shared_ptr<ConstantFunction> m_yold;
    std::shared_ptr<IdentityFunction> m_ynew;
    std::shared_ptr<ConstantFunction> m_vecf;
  public:
    Crank(std::shared_ptr<NonlinearFunction> rhs) 
    : TimeStepper(rhs)
    {
      m_tau = std::make_shared<Parameter>(0.0);
      m_yold = std::make_shared<ConstantFunction>(rhs->dimX());
      m_vecf = std::make_shared<ConstantFunction>(rhs->dimX());
      m_ynew = std::make_shared<IdentityFunction>(rhs->dimX());
    }

    void doStep(double tau, VectorView<double> y) override
    {
      m_yold->set(y);
      this->m_rhs->evaluate(y, m_vecf->get());
      m_tau->set(tau);

      auto m_equ = m_ynew - m_yold - tau / 2.0 * m_vecf - tau / 2.0 * m_rhs;

      NewtonSolver(m_equ, y);
    }
  };


  

}


#endif
