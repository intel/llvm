// RUN: clang++ -fsycl -fsycl-device-only -O0 -w -emit-mlir %s -o - | FileCheck %s --check-prefixes=CHECK
// RUN: clang++ -fsycl -fsycl-device-only -O0 -w -S -emit-llvm -fsycl-targets=spir64-unknown-unknown-syclmlir %s -o - | FileCheck %s --check-prefixes=CHECK-LLVM

// The test checks that:
// 1. !sycl_explicit_simd metadata is generated for functions
// 2. !intel_reqd_sub_group_size 1 is added to explicit SIMD
//    kernel

template <typename name, typename Func>
void kernel(const Func &f) __attribute__((sycl_kernel)) {
  f();
}

void bar() {
  kernel<class MyKernel>([=]() __attribute__((sycl_explicit_simd)){});
  // CHECK-LABEL: @_ZTSZ3barvE8MyKernel()
  // CHECK-SAME:                          intel_reqd_sub_group_size = 1 : i32
  // CHECK-SAME:                          sycl_explicit_simd

  // CHECK-LLVM: @_ZTSZ3barvE8MyKernel()
  // CHECK-LLVM-SAME:                     !intel_reqd_sub_group_size ![[REQD_SIZE:[0-9]+]]
  // CHECK-LLVM-SAME:                     !sycl_explicit_simd ![[EMPTY:[0-9]+]]

  kernel<class MyEsimdKernel>([=]() [[intel::sycl_explicit_simd]]{});
  // CHECK-LABEL: @_ZTSZ3barvE13MyEsimdKernel()
  // CHECK-SAME:                          intel_reqd_sub_group_size = 1 : i32
  // CHECK-SAME:                          sycl_explicit_simd

  // CHECK-LLVM: @_ZTSZ3barvE13MyEsimdKernel()
  // CHECK-LLVM-SAME:                     !intel_reqd_sub_group_size ![[REQD_SIZE:[0-9]+]]
  // CHECK-LLVM-SAME:                     !sycl_explicit_simd ![[EMPTY:[0-9]+]]
}

// CHECK-LLVM: ![[REQD_SIZE]] = !{i32 1}
// CHECK-LLVM: ![[EMPTY]] = !{}
