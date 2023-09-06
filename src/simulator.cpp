#include "simulator.hpp"

#include <utility>
#include <cstdlib>
#include <cmath>

#include <Eigen/Geometry>


cart_pole_simulator::cart_pole_simulator (std::mt19937 &rng) {
  initialize(rng);
}


void cart_pole_simulator::initialize(std::mt19937 &rng) {
  position = (((double)rng())/rng.max())*(MAX_START_POSITION - MIN_START_POSITION) + MIN_START_POSITION;
  velocity = (((double)rng())/rng.max())*(MAX_START_VELOCITY - MIN_START_VELOCITY) + MIN_START_VELOCITY;
  angle = (((double)rng())/rng.max())*(MAX_START_POSITION - MIN_START_POSITION) + MIN_START_POSITION;
  angular_velocity = (((double)rng())/rng.max())*(MAX_START_ANGULAR_VELOCITY - MIN_START_ANGULAR_VELOCITY)\
                                                  + MIN_START_ANGULAR_VELOCITY;
  done = false;

  //update(0.0);
}


void cart_pole_simulator::update(double dt) {
  double force = FORCE_MAG*current_action.thrust;
  double sin_theta = sin(angle);
  double cos_theta = cos(angle);

  double temp = (force + POLE_MASS_LENGTH*angular_velocity*angular_velocity*sin_theta) / TOTAL_MASS;
  double angular_acceleration = (GRAVITY*sin_theta - cos_theta*temp)\
                              / (LENGTH*(4.0/3.0 - MASS_POLE*cos_theta*cos_theta/TOTAL_MASS));
  double acceleration = temp - POLE_MASS_LENGTH*angular_acceleration*cos_theta/TOTAL_MASS;

  if (EULER_KINEMATICS_INTEGRATOR) {
    position = position + velocity*dt;
    velocity = velocity + acceleration*dt;
    angle = angle + angular_velocity*dt;
    angular_velocity = angular_velocity + angular_acceleration*dt;
  } else {
    velocity = velocity + acceleration*dt;
    position = position + velocity*dt;
    angular_velocity = angular_velocity + angular_acceleration*dt;
    angle = angle + angular_velocity*dt;
  }
  
  done = (position < -THESHOLD_POSITION) || (position > THESHOLD_POSITION) ||\
          (angle < -THESHOLD_ANGLE) || (angle > THESHOLD_ANGLE);
}

void cart_pole_simulator::set_action(const action& new_action) {
  current_action.thrust = std::max (-MAX_THRUST(), std::min(MAX_THRUST(), new_action.thrust));
}

VectorXd cart_pole_simulator::get_state() const {
  VectorXd state(4);
  state(0) = position;
  state(1) = velocity;
  state(2) = angle;
  state(3) = angular_velocity;
  return state;
}