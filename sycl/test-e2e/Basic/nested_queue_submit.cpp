// RUN: %{build} -I . -o %t.out
// RUN: %{run} %t.out

#include <cstdlib>
#include <sycl/sycl.hpp>

void nestedSubmit() {
  uint32_t n = 1024;
  float *ptr = (float *)malloc(n * sizeof(float));
  sycl::queue q{};
  {
    sycl::buffer<float> buf(ptr, sycl::range<1>{n});
    q.submit([&](sycl::handler &h) {
      auto acc = buf.get_access<sycl::access::mode::write>(h);
      q.parallel_for<class zero>(sycl::range<1>{n},
                                 [=](sycl::id<1> i) { acc[i] = float(0.0); });
    });
  }
  free(ptr);
}

int main() {
  try {
    nestedSubmit();
  } catch (const sycl::exception &e) {
    assert(e.code() == sycl::errc::invalid && "Invalid error code");
    assert(std::string(e.what()) ==
               "Calls to sycl::queue::submit cannot be nested. Command group "
               "function objects should use the sycl::handler API instead." &&
           "Invalid e.what() string");
  }
  std::cout << "test passed" << std::endl;
  return EXIT_SUCCESS;
}
