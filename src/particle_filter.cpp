/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <sstream>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, const double std[]) {
    if (!is_initialized) {
        normal_distribution<double>
                dist_x(x, std[0]),
                dist_y(y, std[1]),
                dist_theta(theta, std[2]);

        for (int i = 0; i < num_particles; i++) {
            Particle particle;
            particle.id = i;
            particle.x = dist_x(gen);
            particle.y = dist_y(gen);
            particle.theta = dist_theta(gen);
            particle.weight = 1.0;

            particles.push_back(particle);
        }

        is_initialized = true;
    }
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    normal_distribution<double>
            norm_x(0, std_pos[0]),
            norm_y(0, std_pos[1]),
            norm_theta(0, std_pos[2]);

    for (auto &particle : particles) {
        if (fabs(yaw_rate) < 0.000001) {
            // avoid division by zero
            particle.x += velocity * delta_t * cos(particle.theta);
            particle.y += velocity * delta_t * sin(particle.theta);
        } else {
            particle.x += velocity / yaw_rate * (sin(particle.theta + yaw_rate * delta_t) - sin(particle.theta));
            particle.y += velocity / yaw_rate * (cos(particle.theta) - cos(particle.theta + yaw_rate * delta_t));
        }
        particle.theta += yaw_rate * delta_t;

        // add random Gaussian noise
        particle.x += norm_x(gen);
        particle.y += norm_y(gen);
        particle.theta += norm_theta(gen);
    }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predictedVec, vector<LandmarkObs> &observations) {
    // Find the predicted measurement that is closest to each observed measurement and assign the
    //   observed measurement to this particular landmark.

    for (auto &observation : observations) {
        observation.id = -1;
        double smallestDistance = numeric_limits<double>::max();
        for (auto &predicted : predictedVec) {
            double distance = dist(predicted, observation);
            if (distance < smallestDistance) {
                smallestDistance = distance;
                observation.id = predicted.id;
            }
        }
        if (observation.id == -1) {
            // this can happen if we made a coding error - or we have no predictions, e.g. if we choose a particle too
            //  far away so it does not see any landmarks in sensor range
            // Here we just crash, but in reality we have to continue without information - which would likely lead to a
            // real car crash because we can't find our place on the map!
            throw runtime_error("could find nearest prediction for observation");
        }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   vector<LandmarkObs> observations, Map map_landmarks) {
    // Update the weights of each particle using a mult-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation
    //   3.33
    //   http://planning.cs.uiuc.edu/node99.html

    for (auto &particle : particles) {
        vector<LandmarkObs> predictedVec = findLandmarksInSensorRange(sensor_range, particle, map_landmarks);
        vector<LandmarkObs> transformedObservations = transformToParticlePosition(observations, particle);
        dataAssociation(predictedVec, transformedObservations);
        particle.weight = calcParticleWeight(particle, predictedVec, transformedObservations, std_landmark);
    }
}

vector<LandmarkObs> ParticleFilter::findLandmarksInSensorRange(double sensor_range, Particle &particle, Map &map) {
    vector<LandmarkObs> landmarksInSensorRange;
    for (auto &landmark : map.landmark_list) {
        if (dist(particle, landmark) < sensor_range) {
            landmarksInSensorRange.push_back(landmark);
        }
    }
    return landmarksInSensorRange;
}

vector<LandmarkObs>
ParticleFilter::transformToParticlePosition(const vector<LandmarkObs> &observations, Particle &particle) {
    vector<LandmarkObs> transformedObservations;
    for (auto &observation : observations) {
        LandmarkObs transformed = {
                observation.id, // original id
                particle.x + cos(particle.theta) * observation.x + sin(particle.theta) * observation.y, // transformed x
                particle.y - sin(particle.theta) * observation.x + cos(particle.theta) * observation.y, // transformed y
        };
        transformedObservations.push_back(transformed);
    }
    return transformedObservations;
}

double ParticleFilter::calcParticleWeight(Particle &particle, vector<LandmarkObs> &predictedVec,
                                          vector<LandmarkObs> &observations, const double std_landmark[]) {
    double weight = 1.0;
    for (auto &observation : observations) {
        LandmarkObs associatedLandmark = findLandmarkById(predictedVec, observation.id);
        double std_x = std_landmark[0],
                std_y = std_landmark[1],
                x = associatedLandmark.x,
                y = associatedLandmark.y,
                ux = observation.x,
                uy = observation.y;

        weight *= 1 / (2 * M_PI * std_x * std_y)
                  * exp(-(pow(x - ux, 2) / (2 * std_x * std_x) + pow(y - uy, 2) / (2 * std_y * std_y)));
    }
    return weight;
}

double getWeight(Particle &particle) { return particle.weight; }

void ParticleFilter::resample() {
    vector<Particle> resampled;

    // get weights from particles
    vector<double> weights;
    weights.resize(particles.size());
    transform(particles.begin(), particles.end(), weights.begin(), getWeight);

    // do the actual resampling: roll the dice for num_particles times to get the exact amount of particles again
    discrete_distribution<int> distribution(weights.begin(), weights.end());
    for (int i = 0; i < num_particles; i++) {
        int index = distribution(gen);
        resampled.push_back(particles[index]);
        resampled[i].id = i;
    }

    // overwrite the particles with the new list
    particles = resampled;
}

Particle ParticleFilter::SetAssociations(Particle particle, vector<int> associations, vector<double> sense_x,
                                         vector<double> sense_y) {
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    //Clear the previous associations
    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();

    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
}

string ParticleFilter::getAssociations(Particle best) {
    vector<int> v = best.associations;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseX(Particle best) {
    vector<double> v = best.sense_x;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseY(Particle best) {
    vector<double> v = best.sense_y;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}

LandmarkObs ParticleFilter::findLandmarkById(vector<LandmarkObs> &landmarks, int id) {
    for (auto &landmark : landmarks) {
        if (landmark.id == id) {
            return landmark;
        }
    }
    throw runtime_error("no landmark found with id=" + to_string(id));
}
