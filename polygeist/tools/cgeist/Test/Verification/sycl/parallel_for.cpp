// RUN: clang++ -fsycl -fsycl-targets=spir64-unknown-unknown-syclmlir -O0 -w %s -o %t.00.out 2>&1 | FileCheck %s --allow-empty --implicit-check-not="{{error|Error}}:"
// RUN: clang++ -fsycl -fsycl-targets=spir64-unknown-unknown-syclmlir -O1 -w %s -o %t.01.out 2>&1 | FileCheck %s --allow-empty --implicit-check-not="{{error|Error}}:"
// RUN: clang++ -fsycl -fsycl-targets=spir64-unknown-unknown-syclmlir -O2 -w %s -o %t.02.out 2>&1 | FileCheck %s --allow-empty --implicit-check-not="{{error|Error}}:"

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

// Test that the kernel named `kernel_parallel_for` is generated with the correct signature.
// LLVM: define weak_odr spir_kernel void {{.*}}kernel_parallel_for(
// LLVM-SAME:  i32 addrspace(1)* {{.*}}, [[RANGE_TY:%"class.sycl::_V1::range.1"]]* noundef byval([[RANGE_TY]]) {{.*}}, [[RANGE_TY]]* noundef byval([[RANGE_TY]]) {{.*}}, [[ID_TY:%"class.sycl::_V1::id.1"]]* noundef byval([[ID_TY]]) {{.*}})

#include <sycl/sycl.hpp>
using namespace sycl;
static constexpr unsigned N = 8;

void host_parallel_for(std::array<int, N> &A) {
  auto q = queue{};
  device d = q.get_device();
  std::cout << "Using " << d.get_info<info::device::name>() << "\n";
  auto range = sycl::range<1>{N};

  {
    auto buf = buffer<int, 1>{A.data(), range};
    q.submit([&](handler &cgh) {
      auto A = buf.get_access<access::mode::write>(cgh);
      cgh.parallel_for<class kernel_parallel_for>(range, [=](sycl::id<1> id) {
        A[id] = id;
      });
    });
  }
}

int main() {
  std::array<int, N> A{0};
  host_parallel_for(A);
  for (unsigned i = 0; i < N; ++i) {
    assert(A[i] == i);
  }
  std::cout << "Test passed" << std::endl;
}
