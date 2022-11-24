// RUN: clang++ -fsycl -fsycl-targets=spir64-unknown-unknown-syclmlir -O0 -w %s -o %t.out 2>&1 | FileCheck %s --allow-empty --implicit-check-not="{{error|Error}}:"

// RUN: clang++ -fsycl -fsycl-device-only -O0 -w -emit-llvm -fsycl-targets=spir64-unknown-unknown-syclmlir %s -o %t.bc

// Test that the LLVMIR generated is verifiable.
// RUN: opt -verify -disable-output < %t.bc

// Verify that LLVMIR generated is translatable to SPIRV.
// RUN: llvm-spirv %t.bc

// Test that all referenced sycl header functions are generated.
// RUN: llvm-dis %t.bc
// RUN: cat %t.ll | FileCheck %s --check-prefix=LLVM --implicit-check-not="declare{{.*}}spir_func"

// Test that the kernel named `kernel_stream_copy` is generated with the correct signature.
// LLVM: define weak_odr spir_kernel void {{.*}}kernel_stream_copy(
// LLVM-SAME:  i32 addrspace(1)* {{.*}}, [[RANGE_TY:%"class.sycl::_V1::range.1"]]* noundef byval([[RANGE_TY]]) {{.*}}, [[RANGE_TY]]* noundef byval([[RANGE_TY]]) {{.*}}, [[ID_TY:%"class.sycl::_V1::id.1"]]* noundef byval([[ID_TY]]) {{.*}})

#include <sycl/sycl.hpp>
using namespace sycl;
static constexpr unsigned N = 16;

template <typename T>
void host_stream_copy(std::array<T, N> &A, std::array<T, N> &B) {
  auto q = queue{};
  device d = q.get_device();
  std::cout << "Using " << d.get_info<info::device::name>() << "\n";
  auto range = sycl::range<1>{N};

  {
    auto bufA = buffer<T, 1>{A.data(), range};
    auto bufB = buffer<T, 1>{B.data(), range};
    q.submit([&](handler &cgh) {
      auto A = bufA.template get_access<access::mode::write>(cgh);
      auto B = bufB.template get_access<access::mode::read>(cgh);
      cgh.parallel_for<class kernel_stream_copy>(range, [=](sycl::id<1> id) {
        A[id] = B[id];
      });
    });
  }
}

int main() {
  std::array<int, N> A{0};
  std::array<int, N> B{0};
  for (unsigned i = 0; i < N; ++i) {
    B[i] = i;
  }
  host_stream_copy(A, B);
  for (unsigned i = 0; i < N; ++i) {
    assert(A[i] == i);
  }
  std::cout << "Test passed" << std::endl;
}
