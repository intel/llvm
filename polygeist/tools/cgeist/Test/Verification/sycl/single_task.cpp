// clang-format off

// TODO: Investigate and remove all "Warning"s.
// RUN: clang++ -fsycl -fsycl-targets=spir64-unknown-unknown-syclmlir %s -o %t.out 2>&1 | FileCheck %s --implicit-check-not="{{warning|error|Error}}:"
// RUN: env SYCL_DEVICE_FILTER=host %t.out | FileCheck %s --check-prefix=HOST
// TODO: Add device run:
// env SYCL_DEVICE_FILTER=cpu %t.out | FileCheck %s --check-prefix=DEVICE
// REQUIRES: linux

// RUN: clang++ -fsycl -fsycl-device-only -emit-llvm -fsycl-targets=spir64-unknown-unknown-syclmlir %s -o %t.bc 2>/dev/null

// Test that the LLVMIR generated is verifiable.
// RUN: opt -verify -disable-output < %t.bc

// Verify that LLVMIR generated is translatable to SPIRV.
// RUN: llvm-spirv %t.bc

// RUN: llvm-dis %t.bc
// RUN: cat %t.ll | FileCheck %s --check-prefix=LLVM

// Test that the kernel named `kernel_single_task` is generated with the correct signature.
// LLVM: define weak_odr spir_kernel void {{.*}}kernel_single_task(
// LLVM-SAME:  i32 addrspace(1)* {{.*}}, [[RANGE_TY:%"class.sycl::_V1::range.1"]]* byval([[RANGE_TY]]) {{.*}}, [[RANGE_TY]]* byval([[RANGE_TY]]) {{.*}}, [[ID_TY:%"class.sycl::_V1::id.1"]]* byval([[ID_TY]]) {{.*}})

// Test that all referenced sycl header functions are generated.
// LLVM-NOT: declare {{.*}} spir_func

// HOST: Using SYCL host device
// HOST-NEXT: A[0]=2

// DEVICE: Using {{.*}} CPU
// DEVICE-NEXT: A[0]=1

#include <sycl/sycl.hpp>

void host_single_task(std::array<int, 1> &A) {
  auto q = sycl::queue{};
  sycl::device d = q.get_device();
  std::cout << "Using " << d.get_info<sycl::info::device::name>() << "\n";

  auto range = sycl::range<1>{1};

  {
    auto buf = sycl::buffer<int, 1>{A.data(), range};
    q.submit([&](sycl::handler &cgh) {
      auto A = buf.get_access<sycl::access::mode::write>(cgh);
      cgh.single_task<class kernel_single_task>([=]() {
#ifdef __SYCL_DEVICE_ONLY__
        A[0] = 1;
#else
        A[0] = 2;
#endif
      });
    });
  }
}

int main() {
  std::array<int, 1> A = {0};
  A[0] = 0;
  host_single_task(A);

  std::cout << "A[0]=" << A[0] << "\n";
}
