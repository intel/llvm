// REQUIRES: aspect-fp64

// DEFINE: %{mathflags} = %if cl_options %{/clang:-fno-fast-math%} %else %{-fno-fast-math%}

// RUN: %{build} --offload-new-driver -fsycl-allow-device-image-dependencies %if target-spir %{ -fsycl-device-lib-jit-link -Wno-deprecated %} %{mathflags} -o %t.out
// RUN: %{run} %t.out

// XFAIL: target-native_cpu
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/20142

#include <cmath>
#include <sycl/detail/core.hpp>

using namespace sycl;

// Check that device lib dependencies are resolved with
// -fsycl-allow-device-image-dependencies.
// TODO this test will become redundant once
// -fsycl-allow-device-image-dependencies is enabled by default.
int main() {
  range<1> Range{1};
  queue q;
  buffer<double, 1> buffer1(Range);
  q.submit([&](sycl::handler &cgh) {
    auto Acc = buffer1.get_access<access::mode::write>(cgh);
    cgh.single_task<class DeviceMathTest>([=]() { Acc[0] = std::acosh(1.0); });
  });
  return 0;
}
