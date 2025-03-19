// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// REQUIRES: cuda || hip
// The CUDA and HIP adapters currently have a maximum size for kernel
// arguments. This test verifies that the adapters properly check that the
// kernel arguments aren't overflowing this maximum size.

#include <sycl/sycl.hpp>
#include <array>

int main() {
  sycl::queue q;

  std::array<float, 1005> data;
  data.fill(0);
  data[0] = 5;

  bool caught = false;
  try {
    bool test{false};
    sycl::buffer<bool, 1> b_test{&test, 1};
    q.submit([&data, &b_test](sycl::handler &cgh) {
       sycl::accessor a_test{b_test, cgh};
       cgh.parallel_for(1005, [data, a_test](sycl::id<1> id) {
         int x = data[(unsigned int)id];
         if (x > 0)
           a_test[0] = true;
       });
     }).wait_and_throw();
  } catch (sycl::exception &e) {
    // The adapters should return an error because the kernel arguments are too
    // large, this error should've been transformed into a SYCL exception by
    // the runtime.
    return 0;
  }

  // If no exception was thrown there may be a silent buffer overflow.
  return 1;
}
