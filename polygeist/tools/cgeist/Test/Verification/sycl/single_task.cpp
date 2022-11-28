// RUN: clang++ -fsycl -fsycl-targets=spir64-unknown-unknown-syclmlir -O0 -w %s -o %t.O0.out 2>&1 | FileCheck %s --allow-empty --implicit-check-not="{{error|Error}}:"
// RUN: clang++ -fsycl -fsycl-targets=spir64-unknown-unknown-syclmlir -O1 -w %s -o %t.O1.out 2>&1 | FileCheck %s --allow-empty --implicit-check-not="{{error|Error}}:"
// RUN: clang++ -fsycl -fsycl-targets=spir64-unknown-unknown-syclmlir -O2 -w %s -o %t.O2.out 2>&1 | FileCheck %s --allow-empty --implicit-check-not="{{error|Error}}:"

// RUN: clang++ -fsycl -fsycl-device-only -O0 -w -emit-llvm -fsycl-targets=spir64-unknown-unknown-syclmlir %s -o %t.O0.bc
// RUN: clang++ -fsycl -fsycl-device-only -O1 -w -emit-llvm -fsycl-targets=spir64-unknown-unknown-syclmlir %s -o %t.O1.bc
// RUN: clang++ -fsycl -fsycl-device-only -O2 -w -emit-llvm -fsycl-targets=spir64-unknown-unknown-syclmlir %s -o %t.O2.bc

// Test that the LLVMIR generated is verifiable.
// RUN: opt -verify -disable-output < %t.O0.bc
// RUN: opt -verify -disable-output < %t.O1.bc
// RUN: opt -verify -disable-output < %t.O2.bc

// Verify that LLVMIR generated is translatable to SPIRV.
// RUN: llvm-spirv %t.O0.bc
// RUN: llvm-spirv %t.O1.bc
// RUN: llvm-spirv %t.O2.bc

// Test that all referenced sycl header functions are generated.
// RUN: llvm-dis %t.O0.bc
// RUN: llvm-dis %t.O1.bc
// RUN: llvm-dis %t.O2.bc
// RUN: cat %t.O0.ll | FileCheck %s --check-prefix=LLVM --implicit-check-not="declare{{.*}}spir_func"
// RUN: cat %t.O1.ll | FileCheck %s --check-prefix=LLVM --implicit-check-not="declare{{.*}}spir_func"
// RUN: cat %t.O2.ll | FileCheck %s --check-prefix=LLVM --implicit-check-not="declare{{.*}}spir_func"

// Test that the kernel named `kernel_single_task` is generated with the correct signature.
// LLVM: define weak_odr spir_kernel void {{.*}}kernel_single_task(
// LLVM-SAME:  i32 addrspace(1)* {{.*}}, [[RANGE_TY:%"class.sycl::_V1::range.1"]]* noundef byval([[RANGE_TY]]) {{.*}}, [[RANGE_TY]]* noundef byval([[RANGE_TY]]) {{.*}}, [[ID_TY:%"class.sycl::_V1::id.1"]]* noundef byval([[ID_TY]]) {{.*}})

#include <sycl/sycl.hpp>
using namespace sycl;

void host_single_task(std::array<int, 1> &A) {
  auto q = queue{};
  device d = q.get_device();
  std::cout << "Using " << d.get_info<info::device::name>() << "\n";

  {
    auto buf = buffer<int, 1>{A.data(), 1};
    q.submit([&](handler &cgh) {
      auto A = buf.get_access<access::mode::write>(cgh);
      cgh.single_task<class kernel_single_task>([=]() {
        A[0] = 1;
      });
    });
  }
}

int main() {
  std::array<int, 1> A = {0};
  host_single_task(A);
  assert(A[0] == 1);
  std::cout << "Test passed" << std::endl;
}
