// RUN: %clangxx -Xclang -fsycl-allow-func-ptr -fsycl %s -o %t.out
// RUN: %RUN_ON_HOST %t.out
// FIXME: This test should use runtime early exit once correct check for
// corresponding extension is implemented

#include <CL/sycl.hpp>

#include <algorithm>
#include <iostream>
#include <vector>

[[intel::device_indirectly_callable]]
extern "C" int add(int A, int B) { return A + B; }

int main() {
  const int Size = 10;
  std::vector<long> A(Size, 1);
  std::vector<long> B(Size, 2);

  cl::sycl::queue Q;
  cl::sycl::device D = Q.get_device();
  cl::sycl::context C = Q.get_context();
  cl::sycl::program P(C);

  P.build_with_kernel_type<class K>();
  cl::sycl::kernel KE = P.get_kernel<class K>();

  auto FptrStorage = cl::sycl::ONEAPI::get_device_func_ptr(&add, "add", P, D);
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

  cl::sycl::buffer<long> BufA(A.data(), cl::sycl::range<1>(Size));
  cl::sycl::buffer<long> BufB(B.data(), cl::sycl::range<1>(Size));

  Q.submit([&](cl::sycl::handler &CGH) {
    auto AccA =
        BufA.template get_access<cl::sycl::access::mode::read_write>(CGH);
    auto AccB = BufB.template get_access<cl::sycl::access::mode::read>(CGH);
    CGH.parallel_for<class K>(
        KE, cl::sycl::range<1>(Size), [=](cl::sycl::id<1> Index) {
          auto Fptr =
              cl::sycl::ONEAPI::to_device_func_ptr<decltype(add)>(FptrStorage);
          AccA[Index] = Fptr(AccA[Index], AccB[Index]);
        });
  });

  auto HostAcc = BufA.get_access<cl::sycl::access::mode::read>();
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
