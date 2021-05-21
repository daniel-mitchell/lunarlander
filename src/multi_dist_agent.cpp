#include "multi_dist_agent.hpp"

#include <limits>
#include <random>
#include <iostream>
#include <boost/math/special_functions/digamma.hpp>

using Eigen::VectorXd;
using Eigen::Array2d;

tile_coder multi_dist_agent::make_tile_coder(double tile_weight_exponent, const std::vector<int>& subspaces) {

    const double PI = boost::math::constants::pi<double>();

    struct {
        bool state_signed, state_bounded;
        double tile_size;
        unsigned int num_tiles, num_offsets;
    } cfg[] =
        { // #signed, bounded, tile_size, num_tiles, num_offsets
            {    false,    true,       5.0,         6,           2 }, // xpos
            {    false,    true,       5.0,         4,           2 }, // ypos
            {     true,    true,       2.0,         4,           4 }, // xvel
            {     true,    true,       2.0,         4,           4 }, // yvel
            {     true,   false,      PI/2,         2,           8 }, // rot
            {     true,    true,      PI/6,         3,           4 }  // rotvel
        };

    const unsigned int state_dim = 6;

    VectorXd tile_size (6);
    VectorXi num_tiles (6);
    VectorXi num_offsets (6);

    for (unsigned int i = 0; i < state_dim; ++i) {

        max_state(i) = cfg[i].tile_size * cfg[i].num_tiles - 1e-8;
        min_state(i) = cfg[i].state_signed ? -max_state(i) : 0.0;

        max_clip_state(i) = cfg[i].state_bounded ? max_state(i) : std::numeric_limits<double>::infinity();
        min_clip_state(i) = cfg[i].state_signed ? -max_clip_state(i) : 0.0;

        num_tiles(i) = cfg[i].num_tiles;
        if (cfg[i].state_signed) num_tiles(i) *= 2;
        if (cfg[i].state_bounded) num_tiles(i) += 1;

        tile_size(i) = cfg[i].tile_size;
        num_offsets(i) = cfg[i].num_offsets;
    }

    return tile_coder (tile_size, num_tiles, num_offsets, subspaces, tile_weight_exponent);

}

multi_dist_agent::multi_dist_agent(double lambda, double alpha_v, double alpha_u, double initial_value,
                     int num_features,
                     double tile_weight_exponent,
                     [[maybe_unused]] bool trunc_normal,
                     const std::vector<int>& subspaces, int seed):
                         max_state(6), min_state(6), max_clip_state(6), min_clip_state(6),
                         tc (make_tile_coder (tile_weight_exponent, subspaces), num_features),
                         lambda(lambda), alphaU(alpha_u), alphaVInit(alpha_v) {
    rng.seed(seed);
    rng2.seed(seed);
    u = VectorXd::Zero(tc.get_num_features()*6);
    v = VectorXd::Constant(tc.get_num_features()*2, initial_value);
    w = VectorXd::Zero(tc.get_num_features()*6);
    alphaU /= (4*tc.get_active_features());
    alphaVInit /= tc.get_active_features();
    alphaV = alphaVInit;
    maxParam /= tc.get_active_features();
    i_episode = 0;
    minAction << 0, -lunar_lander_simulator::MAX_RCS();
    maxAction << lunar_lander_simulator::MAX_THRUST(), lunar_lander_simulator::MAX_RCS();
    //
    // returns = {0};
    // returnAmount = 0;
    // rBar = 0;
    // x = VectorXi::Zero(1);
    // eu = VectorXd::Zero(1);
    // ev = VectorXd::Zero(1);
    // chosenDist = false;
    // action = lunar_lander_simulator::action(0,0);
    // alpha = Array2d::Zero();
    // beta = Array2d::Zero();
}

double multi_dist_agent::sumIndices(VectorXd vec, VectorXi ind, int offset/*=0*/) {
    double s = 0;
    for (int i = 0; i < ind.rows(); i++) {
        s += vec[ind[i] + offset];
    }
    return s;
}

lunar_lander_simulator::action multi_dist_agent::computeAction() {
    selectAction();
    double scaledThrust = action.thrust*(maxAction[0] - minAction[0]) + minAction[0];
    double scaledRcs;
    if (chosenDist) {
        scaledRcs = action.rcs*maxAction[1];
    } else {
        scaledRcs = action.rcs*minAction[1];
    }
    return lunar_lander_simulator::action(scaledThrust, scaledRcs);
}

double multi_dist_agent::sampleBeta(double alphaa, double betaa) {
    gamma_distribution<double> gamma1(alphaa, 1);
    gamma_distribution<double> gamma2(betaa, 1);
    double sample1 = gamma1(rng);
    double sample2 = gamma2(rng);
    //Sampling z from beta(a, b) is z = (x/(x+y)) where x is sampled from gamma(a,1) and y from gamma(b,1)
    return sample1/(sample1+sample2);
}

void multi_dist_agent::selectAction() {
    if (rng2()/rng2.max() < epsilon) {
        chosenDist = (rng2()/rng2.max() < 0.5);
    } else {
        chosenDist = (sumIndices(v, x) < sumIndices(v, x, tc.get_num_features()));
    }
    int distOffset = chosenDist ? 2*tc.get_num_features() : 0;
    alpha[0] = exp(sumIndices(u, x)) + 1;
    beta[0] = exp(sumIndices(u, x, tc.get_num_features())) + 1;
    action.thrust = sampleBeta(alpha[0], beta[0]);
    alpha[1] = exp(sumIndices(u, x, 2*tc.get_num_features() + distOffset)) + 1;
    beta[1] = exp(sumIndices(u, x, 3*tc.get_num_features() + distOffset)) + 1;
    action.rcs = sampleBeta(alpha[1], beta[1]);
}

lunar_lander_simulator::action multi_dist_agent::initialize([[maybe_unused]] std::mt19937& rng, VectorXd state) {
    i_episode++;
    if (printOutput) {
        cout << "Starting Episode " << i_episode << endl;
    }
    x = tc.indices(state);
    returnAmount = 0;
    rBar = 0;
    eu = VectorXd::Zero(tc.get_num_features()*6);
    ev = VectorXd::Zero(tc.get_num_features()*2);
    if (alphaDecay) {
        alphaV = alphaVInit*(log(1+alphaDecay)/log(1+alphaDecay*i_episode));
    }
    return computeAction();
}

VectorXd multi_dist_agent::computeGradLog() {
    VectorXd gradLog = VectorXd::Zero(6*tc.get_num_features());
    double alphaThrust = log(action.thrust) + boost::math::digamma(alpha[0] + beta[0])\
                            - boost::math::digamma(alpha[0])*(alpha[0] - 1);
    double betaThrust = log(1 - action.thrust) + boost::math::digamma(alpha[0] + beta[0])\
                            - boost::math::digamma(beta[0])*(beta[0] - 1);
    double alphaRcs = log(action.rcs) + boost::math::digamma(alpha[1] + beta[1])\
                            - boost::math::digamma(alpha[1])*(alpha[1] - 1);
    double betaRcs = log(1 - action.rcs) + boost::math::digamma(alpha[1] + beta[1])\
                            - boost::math::digamma(beta[1])*(beta[1] - 1);
    int offset = tc.get_num_features()*(chosenDist ? 4 : 2);
    for (int i = 0; i < x.rows(); i++) {
        gradLog[x[i]] = alphaThrust;
        gradLog[x[i] + tc.get_num_features()] = betaThrust;
        gradLog[x[i] + offset] = alphaRcs;
        gradLog[x[i] + offset + tc.get_num_features()] = betaRcs;
    }
    return gradLog;
}

lunar_lander_simulator::action multi_dist_agent::update([[maybe_unused]] std::mt19937& rng, VectorXd state,
                                        double reward, [[maybe_unused]] bool terminal, [[maybe_unused]] bool learn) {
    returnAmount += reward;
    VectorXi xPrime = tc.indices(state);
    bool chosenDistPrime = sumIndices(v, xPrime) < sumIndices(v, xPrime, tc.get_num_features());

    // Update Values
    double delta = reward - rBar + gamma*(sumIndices(v, xPrime) +\
            sumIndices(v, xPrime, chosenDistPrime ? 2*tc.get_num_features() : tc.get_num_features())) -\
            (sumIndices(v, x) + sumIndices(v, x, chosenDist ? 2*tc.get_num_features() : tc.get_num_features()));
    rBar += alphaR*delta;
    ev *= gamma*lambda;
    for (int i = 0; i < x.rows(); i++) {
        ev[x[i]]++;
        ev[x[i] + chosenDist ? 2*tc.get_num_features() : tc.get_num_features()]++;
    }
    v += alphaV*delta*ev;

    //Compute gradLog
    VectorXd gradLog = computeGradLog();
    Array2d variance;
    VectorXd varVec(6*tc.get_num_features());
    if (s) {
        variance = alpha*beta/((alpha + beta)*(alpha + beta)*(alpha + beta + 1));
        VectorXd varVecThrust = VectorXd::Constant(tc.get_num_features(), variance[0]);
        VectorXd varVecRcs = VectorXd::Constant(tc.get_num_features(), variance[1]);
        varVec << varVecThrust, varVecThrust, varVecRcs, varVecRcs, varVecRcs, varVecRcs;
    }

    //Update Parameters;
    eu = gamma*lambda*eu + gradLog;
    if (inac) {
        w += alphaV*(delta*eu - gradLog.dot(gradLog)*w);
        if (s) {
            u += alphaU*w.cwiseProduct(varVec);
        } else {
            u += alphaU*w;
        }
    } else {
        if (s) {
            u += alphaU*delta*eu.cwiseProduct(varVec);
        } else {
            u += alphaU*delta*eu;
        }
    }
    u = u.cwiseMin(maxParam);
    x = xPrime;

    return computeAction();
}
