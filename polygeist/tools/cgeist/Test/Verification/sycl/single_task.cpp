// RUN: clang++ -Xcgeist --use-opaque-pointers=1 -fsycl -fsycl-targets=spir64-unknown-unknown-syclmlir -O0 -w %s -o %t.O0.out 2>&1 | FileCheck %s --allow-empty --implicit-check-not="{{error|Error}}:"
// RUN: clang++ -Xcgeist --use-opaque-pointers=1 -fsycl -fsycl-targets=spir64-unknown-unknown-syclmlir -O1 -w %s -o %t.O1.out 2>&1 | FileCheck %s --allow-empty --implicit-check-not="{{error|Error}}:"
// RUN: clang++ -Xcgeist --use-opaque-pointers=1 -fsycl -fsycl-targets=spir64-unknown-unknown-syclmlir -O2 -w %s -o %t.O2.out 2>&1 | FileCheck %s --allow-empty --implicit-check-not="{{error|Error}}:"
// RUN: clang++ -Xcgeist --use-opaque-pointers=1 -fsycl -fsycl-targets=spir64-unknown-unknown-syclmlir -O3 -w %s -o %t.O3.out 2>&1 | FileCheck %s --allow-empty --implicit-check-not="{{error|Error}}:"
// RUN: clang++ -Xcgeist --use-opaque-pointers=1 -fsycl -fsycl-targets=spir64-unknown-unknown-syclmlir -Ofast -w %s -o %t.Ofast.out 2>&1 | FileCheck %s --allow-empty --implicit-check-not="{{error|Error}}:"

// RUN: clang++ -Xcgeist --use-opaque-pointers=1 -fno-sycl-instrument-device-code -fsycl -fsycl-device-only -O0 -w -emit-llvm -fsycl-targets=spir64-unknown-unknown-syclmlir %s -o %t.bc

// Test that the LLVMIR generated is verifiable.
// RUN: opt -passes=verify -disable-output < %t.bc

// Verify that LLVMIR generated is translatable to SPIRV.
// RUN: llvm-spirv %t.bc

// Test that all referenced sycl header functions are generated.
// RUN: llvm-dis %t.bc
// RUN: cat %t.ll | FileCheck %s --check-prefix=LLVM --implicit-check-not="declare{{.*}}spir_func"

// Test that the kernel named `kernel_single_task` is generated with the correct signature.
// LLVM: define weak_odr spir_kernel void {{.*}}kernel_single_task(
// LLVM-SAME:  ptr addrspace(1) {{.*}}, ptr noundef byval([[RANGE_TY:%"class.sycl::_V1::range.1"]]) {{.*}}, ptr noundef byval([[RANGE_TY]]) {{.*}}, ptr noundef byval([[ID_TY:%"class.sycl::_V1::id.1"]]) {{.*}})

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
