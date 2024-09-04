// This test checks the output for the clang-offload-wrapper for the Native CPU
// target:
// RUN: %clangxx -fsycl-device-only -fsycl-targets=native_cpu %s -o %t.bc
// RUN: sycl-post-link -properties -emit-param-info -symbols -emit-exported-symbols -O2 -spec-const=native -device-globals -o %t.table %t.bc
// RUN: clang-offload-wrapper -o=%t_wrap.bc -host=x86_64-unknown-linux-gnu -target=native_cpu -kind=sycl -batch %t.table
// RUN: llvm-dis %t_wrap.bc -o - | FileCheck %s

#include <sycl/sycl.hpp>

#include <array>
#include <iostream>

constexpr sycl::access::mode sycl_read = sycl::access::mode::read;
constexpr sycl::access::mode sycl_write = sycl::access::mode::write;

class SimpleVadd;

int main() {
  const size_t N = 4;
  std::array<int, N> A = {{1, 2, 3, 4}}, B = {{2, 3, 4, 5}}, C{{0, 0, 0, 0}};
  sycl::queue deviceQueue;
  sycl::range<1> numOfItems{N};
  sycl::buffer<int, 1> bufferA(A.data(), numOfItems);
  sycl::buffer<int, 1> bufferB(B.data(), numOfItems);
  sycl::buffer<int, 1> bufferC(C.data(), numOfItems);

  deviceQueue
      .submit([&](sycl::handler &cgh) {
        auto accessorA = bufferA.get_access<sycl_read>(cgh);
        auto accessorB = bufferB.get_access<sycl_read>(cgh);
        auto accessorC = bufferC.get_access<sycl_write>(cgh);

        auto kern = [=](sycl::id<1> wiID) {
          accessorC[wiID] = accessorA[wiID] + accessorB[wiID];
        };
        cgh.parallel_for<class SimpleVadd>(numOfItems, kern);
      })
      .wait();

  for (unsigned int i = 0; i < N; i++) {
    std::cout << "C[" << i << "] = " << C[i] << "\n";
    if (C[i] != A[i] + B[i]) {
      std::cout << "The results are incorrect (element " << i << " is " << C[i]
                << "!\n";
      return 1;
    }
  }
  std::cout << "The results are correct!\n";
  return 0;
}
// Check that the kernel name is added as a string in wrapper module
// CHECK: [[KERNELNAME:@__ncpu_function_name.[0-9]*]] = internal unnamed_addr constant [17 x i8] c"_ZTS10SimpleVadd\00"
// Check that the string for the end entry is added
// CHECK: @__ncpu_end_str = internal unnamed_addr constant [16 x i8] c"__nativecpu_end\00"
// Check that the array of declarations for Native CPU is added to the module,
// and it contains the entry for the kernel and it's terminated by the end entry
// CHECK: @__sycl_native_cpu_decls = internal constant [{{[0-9]*}} x %__nativecpu_entry] [{{.*}} %__nativecpu_entry { ptr [[KERNELNAME]], ptr @_ZTS10SimpleVadd.SYCLNCPU }, %__nativecpu_entry { ptr @__ncpu_end_str, ptr null }]
// Check that the declaration for the kernel is added for the wrapper module
// CHECK-DAG: declare void @_ZTS10SimpleVadd.SYCLNCPU(ptr, ptr)
