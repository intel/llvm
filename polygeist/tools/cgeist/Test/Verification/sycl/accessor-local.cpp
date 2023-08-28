// RUN: clang++  -fsycl -fsycl-targets=spir64-unknown-unknown-syclmlir -O0 -w %s -o %t.O0.out 2>&1 | FileCheck %s --allow-empty --implicit-check-not="{{error|Error}}:"
// RUN: clang++  -fsycl -fsycl-targets=spir64-unknown-unknown-syclmlir -O1 -w %s -o %t.O1.out 2>&1 | FileCheck %s --allow-empty --implicit-check-not="{{error|Error}}:"
// RUN: clang++  -fsycl -fsycl-targets=spir64-unknown-unknown-syclmlir -O2 -w %s -o %t.O2.out 2>&1 | FileCheck %s --allow-empty --implicit-check-not="{{error|Error}}:"
// RUN: clang++  -fsycl -fsycl-targets=spir64-unknown-unknown-syclmlir -O3 -w %s -o %t.O3.out 2>&1 | FileCheck %s --allow-empty --implicit-check-not="{{error|Error}}:"
// RUN: clang++  -fsycl -fsycl-targets=spir64-unknown-unknown-syclmlir -Ofast -w %s -o %t.Ofast.out 2>&1 | FileCheck %s --allow-empty --implicit-check-not="{{error|Error}}:"

// RUN: clang++  -fsycl -fsycl-device-only -O0 -w -emit-llvm -fsycl-targets=spir64-unknown-unknown-syclmlir %s -o %t.bc

// Test that the LLVMIR generated is verifiable.
// RUN: opt -passes=verify -disable-output < %t.bc

// Verify that LLVMIR generated is translatable to SPIRV.
// RUN: llvm-spirv %t.bc

// Test that all referenced sycl header functions are generated.
// RUN: llvm-dis %t.bc
// RUN: cat %t.ll | FileCheck %s --check-prefix=LLVM 

// Test that the kernel named `kernel_stream_triad` is generated with the correct signature.
// LLVM: define weak_odr spir_kernel void {{.*}}acc_local_kernel
// LLVM-SAME: (ptr addrspace(3) noundef align 4 %0, ptr noundef byval([[RANGE_TY:%"class.sycl::_V1::range.1"]]) {{.*}}, ptr noundef byval([[RANGE_TY]]) {{.*}}, ptr noundef byval([[ID_TY:%"class.sycl::_V1::id.1"]]) {{.*}}, ptr addrspace(1) noundef align 4 {{.*}}, ptr noundef byval([[RANGE_TY]]) {{.*}}, ptr noundef byval([[RANGE_TY]]) {{.*}}, ptr noundef byval([[ID_TY]]) {{.*}})

#include <vector>

#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
  auto q = sycl::queue{};
  auto range = sycl::range<1>{256};
  std::vector<int> data(256, 0);
  {
    auto buf = buffer{data};
    q.submit([&](sycl::handler &cgh) {
      auto acc = buf.get_access<sycl::access::mode::read_write>(cgh);
      sycl::accessor<int, 1, sycl::access::mode::read_write, target::local>
          local{8, cgh};
      cgh.parallel_for<class acc_local_kernel>(
          nd_range<1>{{256}, {8}}, [=](nd_item<1> item) {
            auto global_id = item.get_global_id();
            auto local_id = item.get_local_id();
            local[local_id] = local_id;
            acc[global_id] = local[local_id];
          });
    });
  }

  for (int i = 0; i < 256; ++i) {
    assert(data[i] == (i % 8));
  }

  return 0;
}
