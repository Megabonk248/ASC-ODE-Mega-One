#ifndef MASS_SPRING_HPP
#define MASS_SPRING_HPP

#include <nonlinfunc.hpp>
#include <timestepper.hpp>

using namespace ASC_ode;

#include <vector.hpp>
#include <array>
using namespace nanoblas;


template <int D>
class Mass
{
public:
  double mass;
  Vec<D> pos;
  Vec<D> vel = 0.0;
  Vec<D> acc = 0.0;
};


template <int D>
class Fix
{
public:
  Vec<D> pos;
};


class Connector
{
public:
  enum CONTYPE { FIX=1, MASS=2 };
  CONTYPE type;
  size_t nr;
};

std::ostream & operator<< (std::ostream & ost, const Connector & con)
{
  ost << "type = " << int(con.type) << ", nr = " << con.nr;
  return ost;
}

class Spring
{
public:
  double length;  
  double stiffness;
  std::array<Connector,2> connectors;
};

class DistanceConstraint
{
public:
  double length;
  std::array<Connector,2> connectors;
};

template <int D>
class MassSpringSystem
{
  std::vector<Fix<D>> m_fixes;
  std::vector<Mass<D>> m_masses;
  std::vector<Spring> m_springs;
  std::vector<DistanceConstraint> m_distance_constraints;
  Vec<D> m_gravity=0.0;
public:
  void setGravity (Vec<D> gravity) { m_gravity = gravity; }
  Vec<D> getGravity() const { return m_gravity; }

  Connector addFix (Fix<D> p)
  {
    m_fixes.push_back(p);
    return { Connector::FIX, m_fixes.size()-1 };
  }

  Connector addMass (Mass<D> m)
  {
    m_masses.push_back (m);
    return { Connector::MASS, m_masses.size()-1 };
  }
  
  size_t addSpring (Spring s) 
  {
    m_springs.push_back (s);
    return m_springs.size()-1;
  }
  
  size_t addDistanceConstraint (DistanceConstraint d) 
  {
    m_distance_constraints.push_back (d);
    return m_distance_constraints.size()-1;
  }

  auto & fixes() { return m_fixes; } 
  auto & masses() { return m_masses; } 
  auto & springs() { return m_springs; }
  auto & distanceConstraints() { return m_distance_constraints; }
  size_t get_state_vec_size() { return m_masses.size() * D + m_distance_constraints.size(); }

  void getState (VectorView<> out_values, VectorView<> out_dvalues, VectorView<> out_ddvalues)
  {
    auto valmat = out_values.asMatrix(m_masses.size(), D);
    auto dvalmat = out_dvalues.asMatrix(m_masses.size(), D);
    auto ddvalmat = out_ddvalues.asMatrix(m_masses.size(), D);

    size_t index = 0;
    for (size_t i = 0; i < m_masses.size(); i++)
      {
        valmat.row(index / 3) = m_masses[i].pos;
        dvalmat.row(index / 3) = m_masses[i].vel;
        ddvalmat.row(index / 3) = m_masses[i].acc;
        index += 3;
      }

    for (size_t i = 0; i < m_distance_constraints.size(); i++) {
        out_values[index] = 0;
        out_dvalues[index] = 0;
        out_ddvalues[index] = 0;
        index += 1;
    }
  }

  void setState (VectorView<> in_values, VectorView<> in_dvalues, VectorView<> in_ddvalues)
  {
    auto valmat = in_values.asMatrix(m_masses.size(), D);
    auto dvalmat = in_dvalues.asMatrix(m_masses.size(), D);
    auto ddvalmat = in_ddvalues.asMatrix(m_masses.size(), D);

    size_t index = 0;
    for (size_t i = 0; i < m_masses.size(); i++)
      {
        m_masses[i].pos = valmat.row(index / 3);
        m_masses[i].vel = dvalmat.row(index / 3);
        m_masses[i].acc = ddvalmat.row(index / 3);
        index += 3;
      }
  }
};

template <int D>
std::ostream & operator<< (std::ostream & ost, MassSpringSystem<D> & mss)
{
  ost << "fixes:" << std::endl;
  for (auto f : mss.fixes())
    ost << f.pos << std::endl;

  ost << "masses: " << std::endl;
  for (auto m : mss.masses())
    ost << "m = " << m.mass << ", pos = " << m.pos << std::endl;

  ost << "springs: " << std::endl;
  for (auto sp : mss.springs())
    ost << "length = " << sp.length << ", stiffness = " << sp.stiffness
        << ", C1 = " << sp.connectors[0] << ", C2 = " << sp.connectors[1] << std::endl;

  ost << "distance constraints: " << std::endl;
  for (auto sp : mss.distanceConstraints())
    ost << "length = " << sp.length
        << ", C1 = " << sp.connectors[0] << ", C2 = " << sp.connectors[1] << std::endl;
  return ost;
}


template <int D>
class MSS_Function : public NonlinearFunction
{
  MassSpringSystem<D> & mss;
public:
  MSS_Function (MassSpringSystem<D> & _mss)
    : mss(_mss) { }

  virtual size_t dimX() const override { return mss.get_state_vec_size(); }
  virtual size_t dimF() const override{ return mss.get_state_vec_size(); }

  /*//calculates the force (gravity + spring force)
  virtual void evaluate (VectorView<double> in_x, VectorView<double> out_f) const override
  {
    out_f = 0.0;

    auto xmat = in_x.asMatrix(mss.masses().size(), D); //positions
    auto fmat = out_f.asMatrix(mss.masses().size(), D); //forces

    //add gravity to forces
    for (size_t i = 0; i < mss.masses().size(); i++)
      fmat.row(i) = mss.masses()[i].mass*mss.getGravity();

    //add spring forces
    for (auto spring : mss.springs())
      {
        auto [c1,c2] = spring.connectors;
        Vec<D> p1, p2;
        if (c1.type == Connector::FIX)
          p1 = mss.fixes()[c1.nr].pos;
        else
          p1 = xmat.row(c1.nr);
        if (c2.type == Connector::FIX)
          p2 = mss.fixes()[c2.nr].pos;
        else
          p2 = xmat.row(c2.nr);

        //add spring forces to force matrix
        double force = spring.stiffness * (norm(p1-p2)-spring.length); //spring force calculated by length
        Vec<D> dir12 = 1.0/norm(p1-p2) * (p2-p1); //normalize directional vector
        if (c1.type == Connector::MASS)
          fmat.row(c1.nr) -= force*dir12; //add force to point 1
        if (c2.type == Connector::MASS)
          fmat.row(c2.nr) += force*dir12; //add the oposite force to point 2
      }

    for (size_t i = 0; i < mss.masses().size(); i++)
      fmat.row(i) *= 1.0/mss.masses()[i].mass;
  }*/

  //calculates the force (gravity + spring force)
  virtual void evaluate (VectorView<double> in_x, VectorView<double> out_f) const override
  {
    out_f = 0.0;

    auto xmat = in_x.asMatrix(mss.masses().size(), D); //positions
    auto fmat = out_f.asMatrix(mss.masses().size(), D); //forces

    //add gravity to forces
    for (size_t i = 0; i < mss.masses().size(); i++)
      fmat.row(i) = mss.masses()[i].mass*mss.getGravity();

    //add spring forces
    for (auto spring : mss.springs())
    {
      auto [c1,c2] = spring.connectors;
      Vec<D> p1, p2;
      if (c1.type == Connector::FIX)
        p1 = mss.fixes()[c1.nr].pos;
      else
        p1 = xmat.row(c1.nr);
      if (c2.type == Connector::FIX)
        p2 = mss.fixes()[c2.nr].pos;
      else
        p2 = xmat.row(c2.nr);

      //add spring forces to force matrix
      double force = spring.stiffness * (norm(p1-p2)-spring.length); //spring force calculated by length
      Vec<D> dir12 = 1.0/norm(p1-p2) * (p2-p1); //normalize directional vector
      if (c1.type == Connector::MASS)
        fmat.row(c1.nr) += force*dir12; //add force to point 1
      if (c2.type == Connector::MASS)
        fmat.row(c2.nr) -= force*dir12; //add the oposite force to point 2
    }

    size_t constraint_index = 0;
    //add the distance constraints
    for (auto distanceConstraint : mss.distanceConstraints()) {
      auto [c1,c2] = distanceConstraint.connectors;
      Vec<D> p1, p2;
      if (c1.type == Connector::FIX)
        p1 = mss.fixes()[c1.nr].pos;
      else
        p1 = xmat.row(c1.nr);
      if (c2.type == Connector::FIX)
        p2 = mss.fixes()[c2.nr].pos;
      else
        p2 = xmat.row(c2.nr);
      
      // dg/dx bzw. Nabla_x g(x)
      auto dir12 = p2 - p1;
      double dist = norm(dir12);
      Vec<D> dir12_norm = 1.0 / dist * dir12; 
      double lambda = in_x[constraint_index + mss.masses().size() * D];

      // add the constraints
      if (c1.type == Connector::MASS) 
        fmat.row(c1.nr) += lambda * dir12_norm;
      if (c2.type == Connector::MASS) 
        fmat.row(c2.nr) -= lambda * dir12_norm;

      // set the constraints rhs function too#
      out_f[constraint_index + mss.masses().size() * D] = dist - distanceConstraint.length; 

      constraint_index += 1;
    }

    // acceleration a=F/m
    for (size_t i = 0; i < mss.masses().size(); i++)
      fmat.row(i) *= 1.0/mss.masses()[i].mass;
  }
  
  
  //calculates the first force derivative
  virtual void evaluateDeriv (VectorView<double> in_x, MatrixView<double> out_df) const override
  {
    out_df = 0.0;
    auto xmat = in_x.asMatrix(mss.masses().size(), D);

    for (auto spring : mss.springs())
    {
      auto [c1,c2] = spring.connectors;
      Vec<D> p1, p2;
      if (c1.type == Connector::FIX)
        p1 = mss.fixes()[c1.nr].pos;
      else
        p1 = xmat.row(c1.nr);
      if (c2.type == Connector::FIX)
        p2 = mss.fixes()[c2.nr].pos;
      else
        p2 = xmat.row(c2.nr);

      Vec<D> dir = p2 - p1;
      double len = norm(dir);
      Vec<D> dir_norm = 1.0 / len * dir;

      double force = spring.stiffness * (len-spring.length);
      if (c1.type == Connector::MASS) {
        out_df.rows(c1.nr * 3, c1.nr * 3 + 3).col(c1.nr * 3) += spring.stiffness * -p1(0) * 1.0 / len * dir_norm; // f' * v_n
        out_df.rows(c1.nr * 3, c1.nr * 3 + 3).col(c1.nr * 3 + 1) += spring.stiffness * -p1(1) * 1.0 / len * dir_norm;
        out_df.rows(c1.nr * 3, c1.nr * 3 + 3).col(c1.nr * 3 + 2) += spring.stiffness * -p1(2) * 1.0 / len * dir_norm;

        out_df.col(c1.nr * 3)[c1.nr * 3] += -force / len;// f * v_n' -> f * v' / len
        out_df.col(c1.nr * 3 + 1)[c1.nr * 3 + 1] += -force / len;
        out_df.col(c1.nr * 3 + 2)[c1.nr * 3 + 2] += -force / len;

        out_df.rows(c1.nr * 3, c1.nr * 3 + 3).col(c1.nr * 3) += -force * -p1(0) * 1.0 / (len * len) * dir_norm;// f * v_n' -> - f * -x_i / len^2 * v_n
        out_df.rows(c1.nr * 3, c1.nr * 3 + 3).col(c1.nr * 3 + 1) += -force * -p1(1) * 1.0 / (len * len) * dir_norm;
        out_df.rows(c1.nr * 3, c1.nr * 3 + 3).col(c1.nr * 3 + 2) += -force * -p1(2) * 1.0 / (len * len) * dir_norm;
      }
      if (c2.type == Connector::MASS) {
        out_df.rows(c2.nr * 3, c2.nr * 3 + 3).col(c2.nr * 3) -= spring.stiffness * p2(0) * 1.0 / len * dir_norm;
        out_df.rows(c2.nr * 3, c2.nr * 3 + 3).col(c2.nr * 3 + 1) -= spring.stiffness * p2(1) * 1.0 / len * dir_norm;
        out_df.rows(c2.nr * 3, c2.nr * 3 + 3).col(c2.nr * 3 + 2) -= spring.stiffness * p2(2) * 1.0 / len * dir_norm;
        
        out_df.col(c2.nr * 3)[c2.nr * 3] -= force / len;
        out_df.col(c2.nr * 3 + 1)[c2.nr * 3 + 1] -= force / len;
        out_df.col(c2.nr * 3 + 2)[c2.nr * 3 + 2] -= force / len;

        out_df.rows(c2.nr * 3, c2.nr * 3 + 3).col(c2.nr * 3) -= -force * p2(0) * 1.0 / (len * len) * dir_norm;
        out_df.rows(c2.nr * 3, c2.nr * 3 + 3).col(c2.nr * 3 + 1) -= -force * p2(1) * 1.0 / (len * len) * dir_norm;
        out_df.rows(c2.nr * 3, c2.nr * 3 + 3).col(c2.nr * 3 + 2) -= -force * p2(2) * 1.0 / (len * len) * dir_norm;
      }
    }

    size_t constraint_index = 0;
    for (auto distanceConstraint : mss.distanceConstraints()) {
      auto [c1,c2] = distanceConstraint.connectors;
      Vec<D> p1, p2;
      if (c1.type == Connector::FIX)
        p1 = mss.fixes()[c1.nr].pos;
      else
        p1 = xmat.row(c1.nr);
      if (c2.type == Connector::FIX)
        p2 = mss.fixes()[c2.nr].pos;
      else
        p2 = xmat.row(c2.nr);
      
      
      Vec<D> dir = p2 - p1;
      double len = norm(dir);
      Vec<D> dir_norm = 1.0 / len * dir;
      double lambda = in_x[constraint_index + mss.masses().size() * D];

      if (c1.type == Connector::MASS) {
        out_df.col(c1.nr * 3)[c1.nr * 3] += -lambda / len;// lambda * v_n' -> lambda * v' / len
        out_df.col(c1.nr * 3 + 1)[c1.nr * 3 + 1] += -lambda / len;
        out_df.col(c1.nr * 3 + 2)[c1.nr * 3 + 2] += -lambda / len;

        out_df.rows(c1.nr * 3, c1.nr * 3 + 3).col(c1.nr * 3) += -lambda * -p1(0) * 1.0 / (len * len) * dir_norm;// lambda * v_n' -> - lambda * -x_i / len^2 * v_n
        out_df.rows(c1.nr * 3, c1.nr * 3 + 3).col(c1.nr * 3 + 1) += -lambda * -p1(1) * 1.0 / (len * len) * dir_norm;
        out_df.rows(c1.nr * 3, c1.nr * 3 + 3).col(c1.nr * 3 + 2) += -lambda * -p1(2) * 1.0 / (len * len) * dir_norm;

        out_df.rows(c1.nr * 3, c1.nr * 3 + 3).col(constraint_index + mss.masses().size() * D) += dir_norm; //lambda' * v_n
      }
      if (c2.type == Connector::MASS) {
        out_df.col(c2.nr * 3)[c2.nr * 3] -= lambda / len;// lambda * v_n' -> lambda * v' / len
        out_df.col(c2.nr * 3 + 1)[c2.nr * 3 + 1] -= lambda / len;
        out_df.col(c2.nr * 3 + 2)[c2.nr * 3 + 2] -= lambda / len;

        out_df.rows(c2.nr * 3, c2.nr * 3 + 3).col(c2.nr * 3) -= -lambda * p2(0) * 1.0 / (len * len) * dir_norm;// lambda * v_n' -> - lambda * -x_i / len^2 * v_n
        out_df.rows(c2.nr * 3, c2.nr * 3 + 3).col(c2.nr * 3 + 1) -= -lambda * p2(1) * 1.0 / (len * len) * dir_norm;
        out_df.rows(c2.nr * 3, c2.nr * 3 + 3).col(c2.nr * 3 + 2) -= -lambda * p2(2) * 1.0 / (len * len) * dir_norm;
        
        out_df.rows(c2.nr * 3, c2.nr * 3 + 3).col(constraint_index + mss.masses().size() * D) -= dir_norm; //lambda' * v_n
      }

      out_df.cols(c1.nr * 3, c1.nr * 3 + 3).row(constraint_index + mss.masses().size() * D) = 1.0 / len * -p1;
      out_df.cols(c2.nr * 3, c2.nr * 3 + 3).row(constraint_index + mss.masses().size() * D) = 1.0 / len * p2;

      constraint_index += 1;
    }

    // acceleration a=F/m
    for (size_t i = 0; i < mss.masses().size(); i++)
      out_df.row(i) *= 1.0/mss.masses()[i].mass;
  }

  /*//calculates the first force derivative
  virtual void evaluateDeriv (VectorView<double> in_x, MatrixView<double> out_df) const override
  {
    // TODO: exact differentiation
    double eps = 1e-8;
    Vector<> xl(dimX()), xr(dimX()), fl(dimF()), fr(dimF());
    for (size_t i = 0; i < dimX(); i++)
      {
        xl = in_x;
        xl(i) -= eps;
        xr = in_x;
        xr(i) += eps;
        evaluate (xl, fl);
        evaluate (xr, fr);
        out_df.col(i) = 1/(2*eps) * (fr-fl);
      }
  }*/
  
};

#endif
