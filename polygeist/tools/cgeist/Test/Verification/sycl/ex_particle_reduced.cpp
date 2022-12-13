
// RUN: clang++ -fsycl -fsycl-targets=spir64-unknown-unknown-syclmlir -O0 -w -c %s -o %t.O0.out 2>&1 | FileCheck %s --allow-empty --implicit-check-not="{{error|Error}}:"
// RUN: clang++ -fsycl -fsycl-device-only -O0 -w -emit-llvm -fsycl-targets=spir64-unknown-unknown-syclmlir %s -o %t.bc

// Test that the LLVMIR generated is verifiable.
// RUN: opt -verify -disable-output < %t.bc

// Verify that LLVMIR generated is translatable to SPIRV.
// RUN: llvm-spirv %t.bc

// Test that all referenced sycl header functions are generated.
// RUN: llvm-dis %t.bc
// RUN: cat %t.ll | FileCheck %s --check-prefix=LLVM --implicit-check-not="declare{{.*}}spir_func"

// Test that the kernel named `kernel_parallel_for_id` is generated with the correct signature.
// LLVM: define weak_odr spir_kernel void {{.*}}kernel_likelihood(
// LLVM-SAME: float addrspace(1)* noundef %0, %"class.sycl::_V1::range.1"* noundef byval(%"class.sycl::_V1::range.1") align 8 %1, 
// LLVM-SAME: %"class.sycl::_V1::range.1"* noundef byval(%"class.sycl::_V1::range.1") align 8 %2, %"class.sycl::_V1::id.1"* noundef byval(%"class.sycl::_V1::id.1") align 8 %3)

#include <sycl/sycl.hpp>
using namespace sycl;
constexpr access::mode sycl_read_write = access::mode::read_write;

void likelyhood(int Nparticles) {
  cpu_selector dev_sel;
  queue q(dev_sel);
  const property_list props = property::buffer::use_host_ptr();
  float *arrayX = (float *)calloc(Nparticles, sizeof(float));
  buffer<float, 1> arrayX_GPU(arrayX, Nparticles, props);

  q.submit([&](handler &cgh) {
    auto arrayX_acc = arrayX_GPU.get_access<sycl_read_write>(cgh);
    cgh.parallel_for<class kernel_likelihood>(nd_range<1>(range<1>(10), range<1>(20)), [=](nd_item<1> item) {
      int i = item.get_global_linear_id();
      arrayX_acc[i] += 1.0;
    });
  });
}
