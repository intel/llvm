// REQUIRES: opencl-aot, cpu
// TODO: Test is failing on Windows with OpenCL, enable back when the issue
// fixed.
// UNSUPPORTED: windows && opencl

// RUN: %clangxx -fsycl -fsycl-targets=spir64_x86_64 %s -o %t.out
// RUN: %t.out

#include <sycl/ext/oneapi/experimental/device_architecture.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental;

int main() {
  std::vector<int> vec(2);
  {
    buffer<int> buf(vec.data(), vec.size());

    queue q(cpu_selector_v);

    // test if_architecture_is
    q.submit([&](handler &cgh) {
      auto acc = buf.get_access<access::mode::read_write>(cgh);
      cgh.single_task([=]() {
        if_architecture_is<architecture::x86_64>([&]() {
          acc[0] = 1;
        }).otherwise([&]() { acc[0] = 0; });
      });
    });

    // test else_if_architecture_is
    q.submit([&](handler &cgh) {
      auto acc = buf.get_access<access::mode::read_write>(cgh);
      cgh.single_task([=]() {
        if_architecture_is<architecture::intel_gpu_dg1>([&]() {
          acc[1] = 0;
        }).else_if_architecture_is<architecture::x86_64>([&]() {
            acc[1] = 2;
          }).otherwise([&]() { acc[1] = 0; });
      });
    });

    // test otherwise
    q.submit([&](handler &cgh) {
      auto acc = buf.get_access<access::mode::read_write>(cgh);
      cgh.single_task([=]() {
        if_architecture_is<architecture::intel_gpu_dg1>([&]() {
          acc[2] = 0;
        }).otherwise([&]() { acc[2] = 3; });
      });
    });

    // test more than one architecture template parameter is passed to
    // if_architecture_is
    q.submit([&](handler &cgh) {
      auto acc = buf.get_access<access::mode::read_write>(cgh);
      cgh.single_task([=]() {
        if_architecture_is<architecture::intel_gpu_dg1, architecture::x86_64>(
            [&]() { acc[3] = 4; })
            .otherwise([&]() { acc[3] = 0; });
      });
    });
  }

  assert(vec[0] == 1);
  assert(vec[1] == 2);
  assert(vec[2] == 3);
  assert(vec[3] == 4);

  return 0;
}
