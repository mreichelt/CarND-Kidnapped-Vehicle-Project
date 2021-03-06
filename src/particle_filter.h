/*
 * particle_filter.h
 *
 * 2D particle filter class.
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#ifndef PARTICLE_FILTER_H_
#define PARTICLE_FILTER_H_

#include "helper_functions.h"
#include <random>

struct Particle {

    int id;
    double x;
    double y;
    double theta;
    double weight;
    std::vector<int> associations;
    std::vector<double> sense_x;
    std::vector<double> sense_y;
};

inline double dist(Particle &p1, LandmarkObs &l2) {
    return dist(p1.x, p1.y, l2.x, l2.y);
}

class ParticleFilter {

    // Number of particles to draw
    unsigned long num_particles{10};

    // Flag, if filter is initialized
    bool is_initialized{false};

    // random engine for re-use
    std::default_random_engine gen;

public:

    // Set of current particles
    std::vector<Particle> particles;

    // Constructor
    // @param M Number of particles
    ParticleFilter() = default;

    // Destructor
    ~ParticleFilter() = default;

    /**
     * init Initializes particle filter by initializing particles to Gaussian
     *   distribution around first position and all the weights to 1.
     * @param x Initial x position [m] (simulated estimate from GPS)
     * @param y Initial y position [m]
     * @param theta Initial orientation [rad]
     * @param std[] Array of dimension 3 [standard deviation of x [m], standard deviation of y [m]
     *   standard deviation of yaw [rad]]
     */
    void init(double x, double y, double theta, const double std[]);

    /**
     * prediction Predicts the state for the next time step
     *   using the process model.
     * @param delta_t Time between time step t and t+1 in measurements [s]
     * @param std_pos[] Array of dimension 3 [standard deviation of x [m], standard deviation of y [m]
     *   standard deviation of yaw [rad]]
     * @param velocity Velocity of car from t to t+1 [m/s]
     * @param yaw_rate Yaw rate of car from t to t+1 [rad/s]
     */
    void prediction(double delta_t, double std_pos[], double velocity, double yaw_rate);

    /**
     * dataAssociation Finds which observations correspond to which landmarks (likely by using
     *   a nearest-neighbors data association).
     * @param predictedVec Vector of predicted landmark observations
     * @param observations Vector of landmark observations
     */
    void dataAssociation(std::vector<LandmarkObs> predictedVec, std::vector<LandmarkObs> &observations);

    /**
     * updateWeights Updates the weights for each particle based on the likelihood of the
     *   observed measurements.
     * @param sensor_range Range [m] of sensor
     * @param std_landmark[] Array of dimension 2 [standard deviation of range [m],
     *   standard deviation of bearing [rad]]
     * @param observations Vector of landmark observations
     * @param map Map class containing map landmarks
     */
    void updateWeights(double sensor_range, double std_landmark[], std::vector<LandmarkObs> observations,
                       Map map_landmarks);

    /**
     * resample Resamples from the updated set of particles to form
     *   the new set of particles.
     */
    void resample();

    /*
     * Set a particles list of associations, along with the associations calculated world x,y coordinates
     * This can be a very useful debugging tool to make sure transformations are correct and assocations correctly connected
     */
    Particle SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x,
                             std::vector<double> sense_y);

    std::string getAssociations(Particle best);

    std::string getSenseX(Particle best);

    std::string getSenseY(Particle best);

    /**
     * initialized Returns whether particle filter is initialized yet or not.
     */
    const bool initialized() const {
        return is_initialized;
    }

    std::vector<LandmarkObs> findLandmarksInSensorRangeAndTransform(double sensor_range, Particle &particle, Map &map);

    double calcParticleWeight(Particle &particle, std::vector<LandmarkObs> &predictedVec,
                              std::vector<LandmarkObs> &observations, const double std_landmark[]);

    LandmarkObs findLandmarkById(std::vector<LandmarkObs> &landmarks, int id);
};


#endif /* PARTICLE_FILTER_H_ */
