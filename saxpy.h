#include <iostream>
#include <cmath>

const unsigned N = (1 << 24);
//const unsigned N = (1 << 5);
const unsigned XVAL = 2.0f;
const unsigned YVAL = 1.0f;
const unsigned AVAL = 3.0f;

typedef float real_t;

static bool saxpy_verify(const real_t *y)
{
    float err = 0.0;
    for (size_t i = 0; i < N; ++i)
	err = err + fabs(y[i] - (AVAL * XVAL + YVAL));

    std::cout << "Errors: " << err << std::endl;
    return err == 0.0;
}

#if __cplusplus >= 201103L || ( defined(_MSC_VER) && _MSC_VER >= 1800 )
#include <chrono>
class saxpy_timer
{
public:
    saxpy_timer() { reset(); }
    void reset() {
	t0_ = std::chrono::high_resolution_clock::now();
    }
    double elapsed(bool reset_timer=false) {
	std::chrono::high_resolution_clock::time_point t =
			std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> time_span =
			std::chrono::duration_cast<std::chrono::duration<double>>(t - t0_);
	if (reset_timer)
	    reset();
	return time_span.count();
    }
    double elapsed_msec(bool reset_timer=false) {
	return elapsed(reset_timer) * 1000;
    }
private:
    std::chrono::high_resolution_clock::time_point t0_;
};
#else
#	error Please provide different timer implementation
#endif
