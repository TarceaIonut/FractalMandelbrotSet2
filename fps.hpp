#ifndef HPP_FPS
#define HPP_FPS

#include <deque>

namespace fps {
    class FPS {
        public:
            double getCurrentFPS();
            FPS(int nrOfChecks);
            FPS();
        private:
            int nrOfChecks;
            std::deque<long long> q;
    };
}

#endif
