///
/// Check if bfloat16 example works using fallback libraries
///

// REQUIRES: opencl-aot, ocloc, cpu, gpu-intel-gen9
// UNSUPPORTED: cuda
// CUDA is not compatible with SPIR.

// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// RUN: %clangxx -fsycl -fsycl-targets=spir64 %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend "-device gen9" %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend "-device *" %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// RUN: %clangxx -fsycl -fsycl-targets=spir64,spir64_gen -Xsycl-target-backend=spir64_gen "-device gen9" %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// RUN: %clangxx -fsycl -fsycl-targets=spir64_x86_64,spir64_gen -Xsycl-target-backend=spir64_gen "-device gen9" %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// RUN: %clangxx -fsycl -fsycl-targets=spir64,spir64_gen -Xsycl-target-backend=spir64_gen "-device pvc" %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out

// RUN: %clangxx -fsycl -fsycl-targets=spir64_x86_64,spir64_gen -Xsycl-target-backend=spir64_gen "-device pvc" %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out

#include <sycl/sycl.hpp>

using namespace sycl;
using sycl::ext::oneapi::bfloat16;

float foo(float a, float b) {
  // Convert from float to bfloat16.
  bfloat16 A{a};
  bfloat16 B{b};

  // Convert A and B from bfloat16 to float, do addition on floating-point
  // numbers, then convert the result to bfloat16 and store it in C.
  bfloat16 C = A + B;

  // Return the result converted from bfloat16 to float.
  return C;
}

int main(int argc, char *argv[]) {
  float data[3] = {7.0f, 8.1f, 0.0f};

  float result_host = foo(7.0f, 8.1f);
  std::cout << "CPU Result = " << result_host << std::endl;
  if (std::abs(15.1f - result_host) > 0.1f) {
    std::cout << "Test failed. Expected CPU Result ~= 15.1" << std::endl;
    return 1;
  }

  queue deviceQueue;
  buffer<float, 1> buf{data, 3};

  deviceQueue.submit([&](handler &cgh) {
    accessor numbers{buf, cgh, read_write};
    cgh.single_task([=]() { numbers[2] = foo(numbers[0], numbers[1]); });
  });

  host_accessor hostOutAcc{buf, read_only};
  float result_device = hostOutAcc[2];
  std::cout << "GPU Result = " << result_device << std::endl;
  if (std::abs(result_host - result_device) > 0.1f) {
    std::cout << "Test failed. CPU Result !~= GPU result" << std::endl;
    return 1;
  }

  return 0;
}
