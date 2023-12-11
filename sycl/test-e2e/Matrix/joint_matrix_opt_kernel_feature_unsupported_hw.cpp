// REQUIRES: gpu-intel-gen12

// RUN: %{build} -o %t.out -DSYCL_EXT_ONEAPI_MATRIX_VERSION=4
// RUN: %{run} %t.out

// Test checks that exception will be thrown in case object of joint_matrix type
// is used on unsupported HW, in this case, on Gen12.

#include <sycl/sycl.hpp>

using namespace sycl::ext::oneapi::experimental::matrix;

int main() {
  sycl::queue q;

  try {
    q.submit([&](sycl::handler &cgh) {
      cgh.single_task([]() {
        joint_matrix<sycl::sub_group, double, use::b, 2, 2, layout::row_major>
            m; // matrix type and sizes do not matter
      });
    });
  } catch (const sycl::exception &e) {
    assert((e.code() == sycl::errc::kernel_not_supported) &&
           (std::string(e.what()) ==
            std::string("no matrix hardware on the target device, joint_matrix "
                        "is not supported")));
  }
  return 0;
}
