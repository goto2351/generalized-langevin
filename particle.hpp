#ifndef PARTICLE_HPP
#define PARTICLE_HPP

namespace generalized_langevin {
    class Particle {
        public:
            Particle(){}
            //座標
            double x;
            double y;
            double z;
            //速度
            double vx[100000];
            double vy[100000];
            double vz[100000];
            double mass;
    };
}//generalized_langevin

#endif