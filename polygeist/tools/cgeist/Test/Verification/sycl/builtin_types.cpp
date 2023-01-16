// RUN: clang++ -fsycl -fsycl-device-only -O0 -w -emit-mlir -o - %s | FileCheck %s

#include <sycl/sycl.hpp>

using namespace sycl;

// CHECK-LABEL: func.func @_Z19opencl_image1d_ro_t14ocl_image1d_ro(
// CHECK:          %arg0: !llvm.ptr<struct<"opencl.image1d_ro_t", opaque>, 1>)
// CHECK-SAME: attributes {[[SPIR_FUNCCC:llvm.cconv = #llvm.cconv<spir_funccc>]], [[LINKEXT:llvm.linkage = #llvm.linkage<external>]],
// CHECK-SAME: [[PASSTHROUGH:passthrough = \["convergent", "mustprogress", "norecurse", "nounwind", \["frame-pointer", "all"\], \["no-trapping-math", "true"\], \["stack-protector-buffer-size", "8"\], \["sycl-module-id", ".*/polygeist/tools/cgeist/Test/Verification/sycl/builtin_types.cpp"\]\]]]} {
SYCL_EXTERNAL void opencl_image1d_ro_t(detail::opencl_image_type<1, access::mode::read, access::target::image>::type var) {}

// CHECK-LABEL: func.func @_Z19opencl_image1d_wo_t14ocl_image1d_wo(
// CHECK:          %arg0: !llvm.ptr<struct<"opencl.image1d_wo_t", opaque>, 1>)
// CHECK-SAME: attributes {[[SPIR_FUNCCC:llvm.cconv = #llvm.cconv<spir_funccc>]], [[LINKEXT:llvm.linkage = #llvm.linkage<external>]],
// CHECK-SAME: [[PASSTHROUGH:passthrough = \["convergent", "mustprogress", "norecurse", "nounwind", \["frame-pointer", "all"\], \["no-trapping-math", "true"\], \["stack-protector-buffer-size", "8"\], \["sycl-module-id", ".*/polygeist/tools/cgeist/Test/Verification/sycl/builtin_types.cpp"\]\]]]} {
SYCL_EXTERNAL void opencl_image1d_wo_t(detail::opencl_image_type<1, access::mode::write, access::target::image>::type var) {}





