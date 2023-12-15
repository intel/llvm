// RUN: %clangxx -%fsycl-host-only -c -ffp-accuracy=high -faltmathlib=SVMLAltMathLibrary -fno-math-errno %s

#include <sycl/sycl.hpp>
using namespace sycl;

int main() {
  queue deviceQueue;
  double Value = 5.;

  deviceQueue.submit([&](handler &cgh) {
    cgh.single_task<class Kernel0>([=]() { double res = std::sin(Value); });
  });

  deviceQueue.submit([&](handler &cgh) {
    cgh.single_task<class Kernel1>([=]() { double res = sycl::sin(Value); });
  });
  return 0;
}
