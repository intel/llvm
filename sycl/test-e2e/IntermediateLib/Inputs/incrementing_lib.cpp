#include <sycl/sycl.hpp>

#if defined(_WIN32)
#define API_EXPORT __declspec(dllexport)
#else
#define API_EXPORT
#endif

#ifndef INC
#define INC 1
#endif

#ifndef CLASSNAME
#define CLASSNAME same
#endif

extern "C" API_EXPORT void performIncrementation(sycl::queue &q,
                                                 sycl::buffer<int, 1> &buf) {
  sycl::range<1> r = buf.get_range();
  q.submit([&](sycl::handler &cgh) {
    auto acc = buf.get_access<sycl::access::mode::write>(cgh);
    cgh.parallel_for<class CLASSNAME>(
        r, [=](sycl::id<1> idx) { acc[idx] += INC; });
  });
}