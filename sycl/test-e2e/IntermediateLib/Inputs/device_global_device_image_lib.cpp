#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

#if defined(_WIN32)
#define API_EXPORT __declspec(dllexport)
#else
#define API_EXPORT
#endif

// Define and export the device_global
__attribute__((visibility("default"))) SYCL_EXTERNAL
    sycl::ext::oneapi::experimental::device_global<int>
        test_global;

// Host function to set the device_global value
extern "C" API_EXPORT void set_test_global(int val) {
  sycl::queue q;
  q.copy(&val, test_global, 1, 0).wait();
}

// Host function to get the device_global value
extern "C" API_EXPORT int get_test_global() {
  sycl::queue q;
  int result = 0;
  q.copy(test_global, &result, 1, 0).wait();
  return result;
}

// Function that reads device_global in a kernel within the library
extern "C" API_EXPORT int read_global_in_lib() {
  sycl::queue q;
  int result = 0;
  int *dev_result = sycl::malloc_device<int>(1, q);

  q.submit([&](sycl::handler &h) {
     h.single_task([=]() {
       dev_result[0] = test_global; // Read in library's own kernel
     });
   }).wait();

  q.copy(dev_result, &result, 1).wait();
  sycl::free(dev_result, q);
  return result;
}
