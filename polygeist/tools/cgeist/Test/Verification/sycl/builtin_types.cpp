// RUN: clang++ -fsycl -fsycl-device-only -O0 -w -emit-mlir -o - %s | FileCheck %s

#include <sycl/sycl.hpp>

using namespace sycl;

// CHECK-LABEL: func.func @_Z13opencl_float2Dv2_f(
// CHECK:         %arg0: memref<?x2xf32> {llvm.noundef})
SYCL_EXTERNAL void opencl_float2(__cl_float2 var) {}

// CHECK-LABEL: func.func @_Z13opencl_float4Dv4_f(
// CHECK:         %arg0: memref<?x4xf32> {llvm.noundef})
SYCL_EXTERNAL void opencl_float4(__cl_float4 var) {}

// CHECK-LABEL: func.func @_Z15scalable_vec2_tN4sycl3_V13vecIfLi2EEE(
// CHECK:         %arg0: memref<?x!sycl_vec_f32_2_> {llvm.align = 8 : i64, llvm.byval = !sycl_vec_f32_2_, llvm.noundef})
SYCL_EXTERNAL void scalable_vec2_t(sycl::vec<float, 2> var) {}

// CHECK-LABEL: func.func @_Z15scalable_vec4_tN4sycl3_V13vecIfLi4EEE(
// CHECK:         %arg0: memref<?x!sycl_vec_f32_4_> {llvm.align = 16 : i64, llvm.byval = !sycl_vec_f32_4_, llvm.noundef})
SYCL_EXTERNAL void scalable_vec4_t(sycl::vec<float, 4> var) {}

// CHECK-LABEL: func.func @_Z8float128g(
// CHECK:         %arg0: f128 {llvm.noundef})
SYCL_EXTERNAL void float128(__float128 var) {}

// CHECK-LABEL: func.func @_Z6int128n(
// CHECK:         %arg0: i128 {llvm.noundef})
SYCL_EXTERNAL void int128(__int128 var) {}

// CHECK-LABEL: func.func @_Z19opencl_image1d_ro_t14ocl_image1d_ro(
// CHECK:         %arg0: !llvm.ptr<struct<"opencl.image1d_ro_t", opaque>, 1>)
SYCL_EXTERNAL void opencl_image1d_ro_t(detail::opencl_image_type<1, access::mode::read, access::target::image>::type var) {}

// CHECK-LABEL: func.func @_Z19opencl_image1d_wo_t14ocl_image1d_wo(
// CHECK:         %arg0: !llvm.ptr<struct<"opencl.image1d_wo_t", opaque>, 1>)
SYCL_EXTERNAL void opencl_image1d_wo_t(detail::opencl_image_type<1, access::mode::write, access::target::image>::type var) {}

// CHECK-LABEL: func.func @_Z25opencl_image1d_array_ro_t20ocl_image1d_array_ro(
// CHECK:         %arg0: !llvm.ptr<struct<"opencl.image1d_array_ro_t", opaque>, 1>)
SYCL_EXTERNAL void opencl_image1d_array_ro_t(detail::opencl_image_type<1, access::mode::read, access::target::image_array>::type var) {}

// CHECK-LABEL: func.func @_Z25opencl_image1d_array_wo_t20ocl_image1d_array_wo(
// CHECK:         %arg0: !llvm.ptr<struct<"opencl.image1d_array_wo_t", opaque>, 1>)
SYCL_EXTERNAL void opencl_image1d_array_wo_t(detail::opencl_image_type<1, access::mode::write, access::target::image_array>::type var) {}

// CHECK-LABEL: func.func @_Z16opencl_sampler_t11ocl_sampler(
// CHECK:         %arg0: !llvm.ptr<struct<"opencl.sampler_t", opaque>, 2>)
SYCL_EXTERNAL void opencl_sampler_t(__ocl_sampler_t var) {}

// CHECK-LABEL: func.func @_Z33opencl_sampled_image_array1d_ro_t38__spirv_SampledImage__image1d_array_ro(
// CHECK:         %arg0: !llvm.ptr<struct<"spirv.SampledImage.image1d_array_ro_t.1", opaque>, 1>)
SYCL_EXTERNAL void opencl_sampled_image_array1d_ro_t(__ocl_sampled_image1d_array_ro_t var) {}

// CHECK-LABEL: func.func @_Z12opencl_vec_tDv4_j(
// CHECK:         %arg0: vector<4xi32> {llvm.noundef})
SYCL_EXTERNAL void opencl_vec_t(__ocl_vec_t<uint32_t, 4> var) {}

// CHECK-LABEL: func.func @_Z14opencl_event_t9ocl_event(
// CHECK:         %arg0: !llvm.ptr<struct<"opencl.event_t", opaque>, 4>)
SYCL_EXTERNAL void opencl_event_t(__ocl_event_t var) {}
