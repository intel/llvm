#include <sycl/detail/core.hpp>

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

#ifdef WITH_DEVICE_GLOBALS
// Using device globals within the shared libraries only
// works if the names do not collide. Note that we cannot
// load a library multiple times if it has a device global.
#define CONCAT_HELPER(a, b) a##b
#define CONCAT(a, b) CONCAT_HELPER(a, b)

using SomeProperties = decltype(sycl::ext::oneapi::experimental::properties{});
sycl::ext::oneapi::experimental::device_global<int, SomeProperties>
    CONCAT(DGVar, CLASSNAME) __attribute__((visibility("default")));

#endif // WITH_DEVICE_GLOBALS

extern "C" API_EXPORT void performIncrementation(sycl::queue &q,
                                                 sycl::buffer<int, 1> &buf) {
  sycl::range<1> r = buf.get_range();
  q.submit([&](sycl::handler &cgh) {
    auto acc = buf.get_access<sycl::access::mode::write>(cgh);
    cgh.parallel_for<class CLASSNAME>(
        r, [=](sycl::id<1> idx) { acc[idx] += INC; });
  });
}
