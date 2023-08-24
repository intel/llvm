// RUN: clang++  -fsycl -fsycl-targets=spir64-unknown-unknown-syclmlir -O0 -w %s -o %t.O0.out 2>&1 | FileCheck %s --allow-empty --implicit-check-not="{{error|Error}}:"
// RUN: clang++  -fsycl -fsycl-targets=spir64-unknown-unknown-syclmlir -O1 -w %s -o %t.O1.out 2>&1 | FileCheck %s --allow-empty --implicit-check-not="{{error|Error}}:"
// RUN: clang++  -fsycl -fsycl-targets=spir64-unknown-unknown-syclmlir -O2 -w %s -o %t.O2.out 2>&1 | FileCheck %s --allow-empty --implicit-check-not="{{error|Error}}:"
// RUN: clang++  -fsycl -fsycl-targets=spir64-unknown-unknown-syclmlir -O3 -w %s -o %t.O3.out 2>&1 | FileCheck %s --allow-empty --implicit-check-not="{{error|Error}}:"
// RUN: clang++  -fsycl -fsycl-targets=spir64-unknown-unknown-syclmlir -Ofast -w %s -o %t.0fast.out 2>&1 | FileCheck %s --allow-empty --implicit-check-not="{{error|Error}}:"

// RUN: clang++  -fsycl -fsycl-device-only -O0 -w -emit-llvm -fsycl-targets=spir64-unknown-unknown-syclmlir %s -o %t.bc

// Test that the LLVMIR generated is verifiable.
// RUN: opt -passes=verify -disable-output < %t.bc

// Verify that LLVMIR generated is translatable to SPIRV.
// RUN: llvm-spirv %t.bc

// Test that all referenced sycl header functions are generated.
// RUN: llvm-dis %t.bc
// RUN: cat %t.ll | FileCheck %s --check-prefix=LLVM 

// Test that the kernel named `kernel_parallel_for_id` is generated with the correct signature.
// LLVM: define weak_odr spir_kernel void {{.*}}kernel_parallel_for_id(
// LLVM-SAME:  ptr addrspace(1) {{.*}}, ptr noundef byval([[RANGE_TY:%"class.sycl::_V1::range.1"]]) {{.*}}, ptr noundef byval([[RANGE_TY]]) {{.*}}, ptr noundef byval([[ID_TY:%"class.sycl::_V1::id.1"]]) {{.*}})

#include <sycl/sycl.hpp>
using namespace sycl;
static constexpr unsigned N = 8;

void parallel_for_id(std::array<int, N> &A, queue q) {
  auto range = sycl::range<1>{N};

  {
    auto buf = buffer<int, 1>{A.data(), range};
    q.submit([&](handler &cgh) {
      auto A = buf.get_access<access::mode::write>(cgh);
      cgh.parallel_for<class kernel_parallel_for_id>(range, [=](id<1> Id) {
        A[Id] = Id;
      });
    });
  }
}

void parallel_for_item(std::array<int, N> &A, queue q) {
  auto range = sycl::range<1>{N};

  {
    auto buf = buffer<int, 1>{A.data(), N};
    q.submit([&](handler &cgh) {
      auto A = buf.get_access<access::mode::write>(cgh);
      cgh.parallel_for<class kernel_parallel_for_item>(range, [=](item<1> Item) {
        id<1> Id = Item.get_id();
        A[Id] = Id;
      });
    });
  }
}

void parallel_for_nd_item(std::array<int, N> &A, queue q) {
  nd_range<1> ndRange(N /*globalSize*/, 2 /*localSize*/);

  {
    auto buf = buffer<int, 1>{A.data(), N};
    q.submit([&](handler &cgh) {
      auto A = buf.get_access<access::mode::write>(cgh);
      cgh.parallel_for<class kernel_parallel_for_nd_item>(ndRange, [=](nd_item<1> NdItem) {
        id<1> Id = NdItem.get_global_id();
        A[Id] = Id;
      });
    });
  }
}

template <typename F>
static void parallel_for(F Func, queue q) {
  std::array<int, N> A{0};
  Func(A, q);
  for (unsigned i = 0; i < N; ++i)
    assert(A[i] == i);
}

int main() {
  auto q = queue{};
  device d = q.get_device();
  std::cout << "Using " << d.get_info<info::device::name>() << "\n";

  parallel_for(parallel_for_id, q);
  parallel_for(parallel_for_item, q);
  parallel_for(parallel_for_nd_item, q);
  std::cout << "Test passed" << std::endl;
}
