// RUN: clang++ -Xcgeist --use-opaque-pointers=1 -fsycl -fsycl-targets=spir64-unknown-unknown-syclmlir -O0 -w %s -o %t.O0.out 2>&1 | FileCheck %s --allow-empty --implicit-check-not="{{error|Error}}:"
// RUN: clang++ -Xcgeist --use-opaque-pointers=1 -fsycl -fsycl-targets=spir64-unknown-unknown-syclmlir -O1 -w %s -o %t.O1.out 2>&1 | FileCheck %s --allow-empty --implicit-check-not="{{error|Error}}:"
// RUN: clang++ -Xcgeist --use-opaque-pointers=1 -fsycl -fsycl-targets=spir64-unknown-unknown-syclmlir -O2 -w %s -o %t.O2.out 2>&1 | FileCheck %s --allow-empty --implicit-check-not="{{error|Error}}:"
// RUN: clang++ -Xcgeist --use-opaque-pointers=1 -fsycl -fsycl-targets=spir64-unknown-unknown-syclmlir -O3 -w %s -o %t.O3.out 2>&1 | FileCheck %s --allow-empty --implicit-check-not="{{error|Error}}:"
// RUN: clang++ -Xcgeist --use-opaque-pointers=1 -fsycl -fsycl-targets=spir64-unknown-unknown-syclmlir -Ofast -w %s -o %t.Ofast.out 2>&1 | FileCheck %s --allow-empty --implicit-check-not="{{error|Error}}:"

// RUN: clang++ -Xcgeist --use-opaque-pointers=1 -fsycl -fsycl-device-only -O0 -w -emit-llvm -fsycl-targets=spir64-unknown-unknown-syclmlir %s -o %t.bc

// Test that the LLVMIR generated is verifiable.
// RUN: opt -passes=verify -disable-output < %t.bc

// Verify that LLVMIR generated is translatable to SPIRV.
// RUN: llvm-spirv %t.bc

// Test that all referenced sycl header functions are generated.
// RUN: llvm-dis %t.bc
// RUN: cat %t.ll | FileCheck %s --check-prefix=LLVM --implicit-check-not="declare{{.*}}spir_func"

// Whitelist declarations of the low-level SPIRV functions used my the math ops.
// LLVM: declare spir_func {{.*}}__spirv_ocl_sqrtf

// Test that the kernel named `kernel_math_funcs` is generated with the correct signature.
// LLVM: define weak_odr spir_kernel void {{.*}}kernel_math_funcs(
// LLVM-SAME:  ptr addrspace(1) {{.*}}, ptr noundef byval([[RANGE_TY:%"class.sycl::_V1::range.1"]]) {{.*}}, ptr noundef byval([[RANGE_TY]]) {{.*}}, ptr noundef byval([[ID_TY:%"class.sycl::_V1::id.1"]]) {{.*}})

#include <climits>
#include <cmath>

#include <sycl/sycl.hpp>
using namespace sycl;

void host_math_funcs(std::array<float, 1> &A) {
  auto q = queue{};
  device d = q.get_device();
  std::cout << "Using " << d.get_info<info::device::name>() << "\n";

  {
    auto buf = buffer<float, 1>{A.data(), 1};
    q.submit([&](handler &cgh) {
      auto A = buf.get_access<access::mode::write>(cgh);
      cgh.single_task<class kernel_math_funcs>([=]() {
        A[0] = sycl::sqrt(2.0f);
      });
    });
  }
}

int main() {
  std::array<float, 1> A = {0.0f};
  host_math_funcs(A);
  assert(std::fabs(A[0] - std::sqrt(2.0f)) < std::numeric_limits<float>::epsilon() );
  std::cout << "Test passed" << std::endl;
}
