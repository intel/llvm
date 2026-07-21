// REQUIRES: arch-intel_gpu_cri
// RUN: %{build} -o %t.out
// RUN: env SYCL_UR_TRACE=2 %{run} %t.out 2>&1 | FileCheck %s

// UNSUPPORTED: target-nvidia || target-amd
// UNSUPPORTED-INTENDED: only supported by backends with CRI driver

#include "../helpers.hpp"
#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

using namespace sycl;
using namespace sycl::detail;
using namespace sycl::ext::intel::experimental;
using namespace sycl::ext::oneapi::experimental;

bool checkResult(const std::vector<float> &A, int Inc) {
  int err_cnt = 0;
  unsigned Size = A.size();

  for (unsigned i = 0; i < Size; ++i) {
    if (A[i] != i + Inc)
      if (++err_cnt < 10)
        std::cerr << "failed at A[" << i << "]: " << A[i] << " != " << i + Inc
                  << "\n";
  }

  if (err_cnt > 0) {
    std::cout << "  pass rate: "
              << ((float)(Size - err_cnt) / (float)Size) * 100.0f << "% ("
              << (Size - err_cnt) << "/" << Size << ")\n";
    return false;
  }
  return true;
}

template <typename T1, typename T2> struct KernelFunctor {
  T1 mPA;
  T2 mProp;
  KernelFunctor(T1 PA, T2 Prop) : mPA(PA), mProp(Prop) {}

  void operator()(id<1> i) const { mPA[i] += 2; }
  auto get(properties_tag) const { return mProp; }
};

int main(void) {
  constexpr unsigned Size = 32;
  constexpr unsigned VL = 16;

  std::vector<float> A(Size);

  for (unsigned i = 0; i < Size; ++i) {
    A[i] = i;
  }

  try {
    buffer<float, 1> bufa(A.data(), range<1>(Size));
    queue q(sycl::gpu_selector_v, exceptionHandlerHelper);

    auto dev = q.get_device();
    std::cout << "Running on " << dev.get_info<sycl::info::device::name>()
              << "\n";

    auto e = q.submit([&](handler &cgh) {
      auto PA = bufa.get_access<access::mode::read_write>(cgh);
      cgh.parallel_for<class SYCLKernelSingleGRF>(Size,
                                                  [=](id<1> i) { PA[i] += 2; });
    });
    e.wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return 2;
  }

  if (checkResult(A, 2)) {
    std::cout << "SingleGRF kernel passed\n";
  } else {
    std::cout << "SingleGRF kernel failed\n";
    return 1;
  }

  try {
    buffer<float, 1> bufa(A.data(), range<1>(Size));
    properties prop{grf_size<512>};
    queue q(sycl::gpu_selector_v, exceptionHandlerHelper);

    auto dev = q.get_device();
    std::cout << "Running on " << dev.get_info<sycl::info::device::name>()
              << "\n";

    auto e = q.submit([&](handler &cgh) {
      auto PA = bufa.get_access<access::mode::read_write>(cgh);
      cgh.parallel_for<class SYCLKernelSpecifiedGRF>(Size,
                                                     KernelFunctor(PA, prop));
    });
    e.wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return 2;
  }

  if (checkResult(A, 4)) {
    std::cout << "SpecifiedGRF kernel passed\n";
  } else {
    std::cout << "SpecifiedGRF kernel failed\n";
    return 1;
  }

  return 0;
}

// CHECK-LABEL: <--- urProgramBuild
// CHECK-SAME: -> UR_RESULT_SUCCESS

// CHECK: <--- urKernelCreate({{.*}}SingleGRF{{.*}}-> UR_RESULT_SUCCESS

// CHECK: <--- urProgramBuild{{.*}}-ze-exp-register-file-size=512
// CHECK-SAME: -> UR_RESULT_SUCCESS

// CHECK: <--- urKernelCreate({{.*}}SpecifiedGRF{{.*}}-> UR_RESULT_SUCCESS
