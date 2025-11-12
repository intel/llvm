// REQUIRES: cuda

// RUN: %clangxx -fsycl -fsycl-targets=nvptx64-nvidia-cuda -S -Xclang -emit-llvm %s -o -| FileCheck %s

#include <sycl/sycl.hpp>

using namespace sycl;

SYCL_EXTERNAL void native_math_cuda(
    accessor<float, 1, access::mode::write, target::device> res_acc,
    accessor<float, 1, access::mode::read, target::device> input_acc) {
  // CHECK: tail call noundef float @llvm.nvvm.cos.approx.f
  res_acc[0] = sycl::native::cos(input_acc[0]);
  // CHECK: tail call noundef float @llvm.nvvm.sin.approx.f
  res_acc[1] = sycl::native::sin(input_acc[0]);
  // CHECK: tail call noundef float @llvm.nvvm.ex2.approx.f
  res_acc[2] = sycl::native::exp2(input_acc[0]);
  // CHECK: tail call noundef float @llvm.nvvm.lg2.approx.f
  res_acc[3] = sycl::native::log2(input_acc[0]);
  // CHECK: tail call noundef float @llvm.nvvm.rsqrt.approx.f
  res_acc[4] = sycl::native::rsqrt(input_acc[0]);
  // CHECK: tail call noundef float @llvm.nvvm.sqrt.approx.f
  res_acc[5] = sycl::native::sqrt(input_acc[0]);
  // CHECK: tail call noundef float @llvm.nvvm.rcp.approx.f
  res_acc[6] = sycl::native::recip(input_acc[0]);
  // CHECK: tail call noundef float @llvm.nvvm.div.approx.f
  res_acc[7] = sycl::native::divide(input_acc[0], input_acc[1]);

  // Functions that use the above builtins:

  // CHECK: tail call float @llvm.nvvm.sin.approx.f
  // CHECK: tail call float @llvm.nvvm.cos.approx.f
  // CHECK: tail call noundef float @llvm.nvvm.div.approx.f
  res_acc[8] = sycl::native::tan(input_acc[0]);
  // CHECK: fmul float {{.*}}, 0x3FF7154760000000
  // CHECK: tail call noundef float @llvm.nvvm.ex2.approx.f
  res_acc[9] = sycl::native::exp(input_acc[0]);
  // CHECK: fmul float {{.*}}, 0x400A934F00000000
  // CHECK: tail call noundef float @llvm.nvvm.ex2.approx.f
  res_acc[10] = sycl::native::exp10(input_acc[0]);
  // CHECK: tail call float @llvm.nvvm.lg2.approx.f
  // CHECK: fmul float {{.*}}, 0x3FE62E4300000000
  res_acc[11] = sycl::native::log(input_acc[0]);
  // CHECK: tail call float @llvm.nvvm.lg2.approx.f
  // CHECK: fmul float {{.*}}, 0x3FD3441360000000
  res_acc[12] = sycl::native::log10(input_acc[0]);

  // CHECK: tail call float @llvm.nvvm.lg2.approx.f
  // CHECK: fmul float {{.*}}, {{.*}}
  // CHECK: tail call noundef float @llvm.nvvm.ex2.approx.f
  res_acc[13] = sycl::native::powr(input_acc[0], input_acc[1]);
};
