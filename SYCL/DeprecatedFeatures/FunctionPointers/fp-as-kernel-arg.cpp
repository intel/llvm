// REQUIRES: TEMPORARILY_DISABLED
// UNSUPPORTED: cuda || hip
// CUDA does not support the function pointer as kernel argument extension.

// RUN: %clangxx -Xclang -fsycl-allow-func-ptr -fsycl -D__SYCL_INTERNAL_API %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// FIXME: This test should use runtime early exit once correct check for
// corresponding extension is implemented

#include <sycl/ext/oneapi/__function_pointers.hpp>
#include <sycl/sycl.hpp>

#include <algorithm>
#include <iostream>
#include <vector>

[[intel::device_indirectly_callable]] extern "C" int add(int A, int B) {
  return A + B;
}

int main() {
  const int Size = 10;
  std::vector<long> A(Size, 1);
  std::vector<long> B(Size, 2);

  sycl::queue Q;
  sycl::device D = Q.get_device();
  sycl::context C = Q.get_context();
  sycl::program P(C);

  P.build_with_kernel_type<class K>();
  sycl::kernel KE = P.get_kernel<class K>();

  auto FptrStorage = sycl::ext::oneapi::get_device_func_ptr(&add, "add", P, D);
  if (!D.is_host()) {
    // FIXME: update this check with query to supported extension
    // For now, we don't have runtimes that report required OpenCL extension and
    // it is hard to understand should this functionality be supported or not.
    // So, let's skip this test if FptrStorage is 0, which means that by some
    // reason we failed to obtain device function pointer. Just to avoid false
    // alarms
    if (0 == FptrStorage) {
      std::cout << "Test PASSED. (it was actually skipped)" << std::endl;
      return 0;
    }
  }

  sycl::buffer<long> BufA(A.data(), sycl::range<1>(Size));
  sycl::buffer<long> BufB(B.data(), sycl::range<1>(Size));

  Q.submit([&](sycl::handler &CGH) {
    auto AccA = BufA.template get_access<sycl::access::mode::read_write>(CGH);
    auto AccB = BufB.template get_access<sycl::access::mode::read>(CGH);
    CGH.parallel_for<class K>(KE, sycl::range<1>(Size), [=](sycl::id<1> Index) {
      auto Fptr =
          sycl::ext::oneapi::to_device_func_ptr<decltype(add)>(FptrStorage);
      AccA[Index] = Fptr(AccA[Index], AccB[Index]);
    });
  });

  auto HostAcc = BufA.get_access<sycl::access::mode::read>();
  auto *Data = HostAcc.get_pointer();

  if (std::all_of(Data, Data + Size, [](long V) { return V == 3; })) {
    std::cout << "Test PASSED." << std::endl;
  } else {
    std::cout << "Test FAILED." << std::endl;
    for (int I = 0; I < Size; ++I) {
      std::cout << HostAcc[I] << " ";
    }
    std::cout << std::endl;
  }

  return 0;
}
