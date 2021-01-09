#ifndef NEWPARTICLE_HPP
#define NEWPARTICLE_HPP

namespace generalized_langevin {
    //更新用の粒子のクラス
    class NewParticle {
        public:
            NewParticle(){}
            //座標
            double x;
            double y;
            double z;
            //速度
            double vx;
            double vy;
            double vz;
            double mass;
    };
}//generalized_langevin

#endif