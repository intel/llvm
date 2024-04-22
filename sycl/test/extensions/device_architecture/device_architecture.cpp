// The goal of this test is to check that new design of
// sycl_ext_oneapi_device_architecture extension can be compiled successfullly.
// During binary run there are some errors, this is expected, so there is no run
// line yet for this test.

// RUN: %clangxx -fsycl -DSYCL_EXT_ONEAPI_DEVICE_ARCHITECTURE_NEW_DESIGN_IMPL %s -o %t.out

#include <sycl/ext/oneapi/experimental/device_architecture.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental;

int main() {
  std::vector<int> vec(4);
  {
    buffer<int> buf(vec.data(), vec.size());

    queue q(gpu_selector_v);

    // test if_architecture_is
    q.submit([&](handler &cgh) {
      auto acc = buf.get_access<access::mode::read_write>(cgh);
      cgh.single_task([=]() {
        if_architecture_is<architecture::intel_gpu_pvc>([&]() {
          acc[0] = 2;
        }).otherwise([&]() { acc[0] = 1; });
      });
    });
  }

  assert(vec[0] == 1);

  return 0;
}
