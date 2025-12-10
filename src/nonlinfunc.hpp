#ifndef NONLINFUNC_H
#define NONLINFUNC_H

#include <cstddef>
#include <memory>

#include <vector.hpp>
#include <matrix.hpp>

namespace ASC_ode
{
  using namespace nanoblas;

  class NonlinearFunction
  {
  public:
    virtual ~NonlinearFunction() = default;
    virtual size_t dimX() const = 0;
    virtual size_t dimF() const = 0;
    virtual void evaluate (VectorView<double> x, VectorView<double> f) const = 0;
    virtual void evaluateDeriv (VectorView<double> x, MatrixView<double> df) const = 0;
  };


  class IdentityFunction : public NonlinearFunction
  {
    size_t m_n;
  public:
    IdentityFunction (size_t n) : m_n(n) { } 
    size_t dimX() const override { return m_n; }
    size_t dimF() const override { return m_n; }
    void evaluate (VectorView<double> x, VectorView<double> f) const override
    {
      f = x;
    }

    void evaluateDeriv (VectorView<double> x, MatrixView<double> df) const override
    {
      df = 0.0;
      df.diag() = 1.0;
    }
  };



  class ConstantFunction : public NonlinearFunction
  {
    Vector<> m_val;
  public:
    ConstantFunction(size_t n) : m_val(n) { }
    ConstantFunction(VectorView<double> val) : m_val(val) { }
    void set(VectorView<double> val) { m_val = val; }
    VectorView<double> get() const { return m_val; }
    size_t dimX() const override { return m_val.size(); }
    size_t dimF() const override { return m_val.size(); }
    void evaluate (VectorView<double> x, VectorView<double> f) const override
    {
      f = m_val;
    }
    void evaluateDeriv (VectorView<double> x, MatrixView<double> df) const override
    {
      df = 0.0;
    }
  };

  
  
  class SumFunction : public NonlinearFunction
  {
    std::shared_ptr<NonlinearFunction> m_fa, m_fb;
    double m_faca, m_facb;
  public:
    SumFunction (std::shared_ptr<NonlinearFunction> fa,
                 std::shared_ptr<NonlinearFunction> fb,
                 double faca, double facb)
      : m_fa(fa), m_fb(fb), m_faca(faca), m_facb(facb) { }

    size_t dimX() const override { return m_fa->dimX(); }
    size_t dimF() const override { return m_fa->dimF(); }
    void evaluate (VectorView<double> x, VectorView<double> f) const override
    {
      m_fa->evaluate(x, f);
      f *= m_faca;
      Vector<> tmp(dimF());
      m_fb->evaluate(x, tmp);
      f += m_facb*tmp;
    }
    void evaluateDeriv (VectorView<double> x, MatrixView<double> df) const override
    {
      m_fa->evaluateDeriv(x, df);
      df *= m_faca;
      Matrix<double> tmp(dimF(), dimX());
      m_fb->evaluateDeriv(x, tmp);
      df += m_facb*tmp;
    }
  };


  inline auto operator- (std::shared_ptr<NonlinearFunction> fa, std::shared_ptr<NonlinearFunction> fb)
  {
    return std::make_shared<SumFunction>(fa, fb, 1, -1);
  }

  inline auto operator+ (std::shared_ptr<NonlinearFunction> fa, std::shared_ptr<NonlinearFunction> fb)
  {
    return std::make_shared<SumFunction>(fa, fb, 1, 1);
  }

  class Parameter 
  {
    double m_value;
  public:
    Parameter(double value) : m_value(value) {}
    double get() const { return m_value; }
    void set(double value) { m_value = value; }
  };

  class ScaleFunction : public NonlinearFunction
  {
    std::shared_ptr<NonlinearFunction> m_fa;
    std::shared_ptr<Parameter> m_fac;
  public:
    ScaleFunction (std::shared_ptr<NonlinearFunction> fa,
                   std::shared_ptr<Parameter> fac)
      : m_fa(fa), m_fac(fac) { }

    size_t dimX() const override { return m_fa->dimX(); }
    size_t dimF() const override { return m_fa->dimF(); }
    void evaluate (VectorView<double> x, VectorView<double> f) const override
    {
      m_fa->evaluate(x, f);
      f *= m_fac->get();
   }

    void evaluateDeriv (VectorView<double> x, MatrixView<double> df) const override
    {
      m_fa->evaluateDeriv(x, df);
      df *= m_fac->get();
    }
  };

  inline auto operator* (std::shared_ptr<Parameter> parama, 
                         std::shared_ptr<NonlinearFunction> f)
  {
    return std::make_shared<ScaleFunction>(f, parama);
  }

  inline auto operator* (double a, std::shared_ptr<NonlinearFunction> f)
  {
    return std::make_shared<Parameter>(a) * f;
  } 




  // fa(fb)
  class ComposeFunction : public NonlinearFunction
  {
    std::shared_ptr<NonlinearFunction> m_fa, m_fb;
  public:
    ComposeFunction (std::shared_ptr<NonlinearFunction> fa,
                     std::shared_ptr<NonlinearFunction> fb)
      : m_fa(fa), m_fb(fb) { }

    size_t dimX() const override { return m_fb->dimX(); }
    size_t dimF() const override { return m_fa->dimF(); }
    void evaluate (VectorView<double> x, VectorView<double> f) const override
    {
      Vector<> tmp(m_fb->dimF());
      m_fb->evaluate (x, tmp);
      m_fa->evaluate (tmp, f);
    }
    void evaluateDeriv (VectorView<double> x, MatrixView<double> df) const override
    {
      Vector<> tmp(m_fb->dimF());
      m_fb->evaluate (x, tmp);

      Matrix<double> jaca(m_fa->dimF(), m_fa->dimX());
      Matrix<double> jacb(m_fb->dimF(), m_fb->dimX());

      m_fb->evaluateDeriv(x, jacb);
      m_fa->evaluateDeriv(tmp, jaca);

      df = jaca*jacb;
    }
  };
  
  
  inline auto Compose (std::shared_ptr<NonlinearFunction> fa, std::shared_ptr<NonlinearFunction> fb)
  {
    return make_shared<ComposeFunction> (fa, fb);
  }
  
  class EmbedFunction : public NonlinearFunction
  {
    std::shared_ptr<NonlinearFunction> m_fa;
    size_t m_firstx, m_dimx, m_firstf, m_dimf;
    size_t m_nextx, m_nextf;
  public:
    EmbedFunction (std::shared_ptr<NonlinearFunction> fa,
                   size_t firstx, size_t dimx,
                   size_t firstf, size_t dimf)
      : m_fa(fa),
        m_firstx(firstx), m_dimx(dimx), m_firstf(firstf), m_dimf(dimf),
        m_nextx(m_firstx+m_fa->dimX()), m_nextf(m_firstf+m_fa->dimF())
    { }

    size_t dimX() const override { return m_dimx; }
    size_t dimF() const override { return m_dimf; }
    void evaluate (VectorView<double> x, VectorView<double> f) const override
    {
      f = 0.0;
      m_fa->evaluate(x.range(m_firstx, m_nextx), f.range(m_firstf, m_nextf));
    }
    void evaluateDeriv (VectorView<double> x, MatrixView<double> df) const override
    {
      df = 0;
      m_fa->evaluateDeriv(x.range(m_firstx, m_nextx),
                        df.rows(m_firstf, m_nextf).cols(m_firstx, m_nextx));
    }
  };

  
  class Projector : public NonlinearFunction
  {
    size_t m_size, m_first, m_next;
  public:
    Projector (size_t size, 
               size_t first, size_t next)
      : m_size(size), m_first(first), m_next(next) { }

    size_t dimX() const override { return m_size; }
    size_t dimF() const override { return m_size; }
    void evaluate (VectorView<double> x, VectorView<double> f) const override
    {
      f = 0.0;
      f.range(m_first, m_next) = x.range(m_first, m_next);
    }
    void evaluateDeriv (VectorView<double> x, MatrixView<double> df) const override
    {
      df = 0.0;
      df.diag().range(m_first, m_next) = 1;
    }
  };

  
  class MultipleFunc : public NonlinearFunction
  {
    std::shared_ptr<NonlinearFunction> func;
    size_t num, fdimx, fdimf;
  public:
    MultipleFunc (std::shared_ptr<NonlinearFunction> _func, int _num)
      : func(_func), num(_num)
    {
      fdimx = func->dimX();
      fdimf = func->dimF();
    }

    virtual size_t dimX() const override { return num * fdimx; } 
    virtual size_t dimF() const override{ return num * fdimf; }
    virtual void evaluate (VectorView<double> x, VectorView<double> f) const override
    {
      for (size_t i = 0; i < num; i++)
        func->evaluate(x.range(i*fdimx, (i+1)*fdimx),
                       f.range(i*fdimf, (i+1)*fdimf));
    }
    virtual void evaluateDeriv (VectorView<double> x, MatrixView<double> df) const override
    {
      df = 0.0;
      for (size_t i = 0; i < num; i++)
        func->evaluateDeriv(x.range(i*fdimx, (i+1)*fdimx),
                            df.rows(i*fdimf, (i+1)*fdimf).cols(i*fdimx, (i+1)*fdimx));
    }
  };


  class MatVecFunc : public NonlinearFunction
  {
    Matrix<> m_a;
    size_t m_n;
  public:
    MatVecFunc (Matrix<> a, size_t n)
      : m_a(a), m_n(n) { }

    virtual size_t dimX() const override { return m_n*m_a.rows(); } 
    virtual size_t dimF() const override { return m_n*m_a.cols(); }
    virtual void evaluate (VectorView<double> x, VectorView<double> f) const override
    {
      MatrixView<double> mx(m_a.cols(), m_n, m_n, x.data());
      MatrixView<double> mf(m_a.rows(), m_n, m_n, f.data());
      mf = m_a * mx;
    }
    virtual void evaluateDeriv (VectorView<double> x, MatrixView<double> df) const override
    {
      df = 0.0;
      for (size_t i = 0; i < m_a.rows(); i++)
        for (size_t j = 0; j < m_a.cols(); j++)
          df.rows(i*m_n, (i+1)*m_n).cols(j*m_n, (j+1)*m_n).diag() = m_a(i,j);
    }
  };

}

#endif
