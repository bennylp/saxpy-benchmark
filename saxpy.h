#define N		(1 << 20)
#define XVAL	2.0f
#define YVAL	1.0f
#define AVAL	3.0f
typedef float real_t;

#if __cplusplus >= 201103L || ( defined(_MSC_VER) && _MSC_VER >= 1800 )
#include <chrono>
class Timer
{
public:
	Timer() { reset(); }
	void reset() {
		t0_ = std::chrono::high_resolution_clock::now();
	}
	double elapsed() {
		std::chrono::high_resolution_clock::time_point t =
				std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> time_span =
				std::chrono::duration_cast<std::chrono::duration<double>>(t - t0_); \
		return time_span.count();
	}
private:
	std::chrono::high_resolution_clock::time_point t0_;
};
#else
#	error Please provide different timer implementation
#endif
