// Check that __invoke__Image* functions are inlined into their callers even at
// -O0. This is required for correct SPIR-V generation: the SPIR-V spec
// (section 2.16.1) requires that OpSampledImage instructions (e.g. the result
// of OpConvertHandleToSampledImageINTEL) are consumed in the same basic block
// as the image operation that uses them (e.g. OpImageSampleExplicitLod).
// Without always_inline, the __invoke__Image* wrappers would not be inlined at
// -O0, placing the handle conversion in a different block from the consumer.

// RUN: %clangxx -O0 -fsycl -fsycl-device-only -fno-discard-value-names -S -emit-llvm -fno-sycl-instrument-device-code -o - %s | FileCheck %s

#include <sycl/sycl.hpp>

using OCLImageTyRead =
    typename sycl::detail::opencl_image_type<2, sycl::access::mode::read,
                                             sycl::access::target::image>::type;

using OCLImageTyWrite =
    typename sycl::detail::opencl_image_type<2, sycl::access::mode::write,
                                             sycl::access::target::image>::type;

using OCLSampledImageTy =
    typename sycl::detail::sampled_opencl_image_type<OCLImageTyRead>::type;

// Test ImageRead - __invoke__ImageRead must be inlined, __spirv_ImageRead must
// appear directly in the caller.
// CHECK-LABEL: define {{.*}} @_Z{{.*}}test_image_read
// CHECK-NOT: call {{.*}} @{{.*}}__invoke__ImageRead
// CHECK: call {{.*}} <4 x float> @{{.*}}__spirv_ImageRead
SYCL_EXTERNAL sycl::float4 test_image_read(OCLImageTyRead img) {
  return __invoke__ImageRead<sycl::float4>(img, sycl::int2(0, 0));
}

// Test ImageWrite - __invoke__ImageWrite must be inlined, __spirv_ImageWrite
// must appear directly in the caller.
// CHECK-LABEL: define {{.*}} @_Z{{.*}}test_image_write
// CHECK-NOT: call {{.*}} @{{.*}}__invoke__ImageWrite
// CHECK: call {{.*}} void @{{.*}}__spirv_ImageWrite
SYCL_EXTERNAL void test_image_write(OCLImageTyWrite img, sycl::float4 val) {
  __invoke__ImageWrite(img, sycl::int2(0, 0), val);
}

// Test ImageFetch - __invoke__ImageFetch must be inlined, __spirv_ImageFetch
// must appear directly in the caller.
// CHECK-LABEL: define {{.*}} @_Z{{.*}}test_image_fetch
// CHECK-NOT: call {{.*}} @{{.*}}__invoke__ImageFetch
// CHECK: call {{.*}} <4 x float> @{{.*}}__spirv_ImageFetch
SYCL_EXTERNAL sycl::float4 test_image_fetch(OCLImageTyRead img) {
  return __invoke__ImageFetch<sycl::float4>(img, sycl::int2(0, 0));
}

// Test SampledImageFetch - __invoke__SampledImageFetch must be inlined,
// __spirv_SampledImageFetch must appear directly in the caller.
// CHECK-LABEL: define {{.*}} @_Z{{.*}}test_sampled_image_fetch
// CHECK-NOT: call {{.*}} @{{.*}}__invoke__SampledImageFetch
// CHECK: call {{.*}} <4 x float> @{{.*}}__spirv_SampledImageFetch
SYCL_EXTERNAL sycl::float4 test_sampled_image_fetch(OCLSampledImageTy img) {
  return __invoke__SampledImageFetch<sycl::float4>(img, sycl::int2(0, 0));
}

// Test ImageReadLod - __invoke__ImageReadLod must be inlined,
// __spirv_ImageSampleExplicitLod must appear directly in the caller.
// CHECK-LABEL: define {{.*}} @_Z{{.*}}test_image_read_lod
// CHECK-NOT: call {{.*}} @{{.*}}__invoke__ImageReadLod
// CHECK: call {{.*}} <4 x float> @{{.*}}__spirv_ImageSampleExplicitLod
SYCL_EXTERNAL sycl::float4 test_image_read_lod(OCLSampledImageTy img) {
  return __invoke__ImageReadLod<sycl::float4>(img, sycl::float2(0.f, 0.f), 0.f);
}

// Test ImageReadGrad - __invoke__ImageReadGrad must be inlined,
// __spirv_ImageSampleExplicitLod must appear directly in the caller.
// CHECK-LABEL: define {{.*}} @_Z{{.*}}test_image_read_grad
// CHECK-NOT: call {{.*}} @{{.*}}__invoke__ImageReadGrad
// CHECK: call {{.*}} <4 x float> @{{.*}}__spirv_ImageSampleExplicitLod
SYCL_EXTERNAL sycl::float4 test_image_read_grad(OCLSampledImageTy img) {
  return __invoke__ImageReadGrad<sycl::float4>(img, sycl::float2(0.f, 0.f),
                                               sycl::float2(0.f, 0.f),
                                               sycl::float2(0.f, 0.f));
}

// Test ImageReadCubemap - __invoke__ImageReadCubemap must be inlined,
// __spirv_ImageSampleCubemap must appear directly in the caller.
// CHECK-LABEL: define {{.*}} @_Z{{.*}}test_image_read_cubemap
// CHECK-NOT: call {{.*}} @{{.*}}__invoke__ImageReadCubemap
// CHECK: call {{.*}} <4 x float> @{{.*}}__spirv_ImageSampleCubemap
SYCL_EXTERNAL sycl::float4 test_image_read_cubemap(OCLSampledImageTy img) {
  return __invoke__ImageReadCubemap<sycl::float4>(img,
                                                  sycl::float3(0.f, 0.f, 0.f));
}

// Test ImageReadSampler - __invoke__ImageReadSampler must be inlined,
// __spirv_ImageSampleExplicitLod must appear directly in the caller.
// CHECK-LABEL: define {{.*}} @_Z{{.*}}test_image_read_sampler
// CHECK-NOT: call {{.*}} @{{.*}}__invoke__ImageReadSampler
// CHECK: call {{.*}} <4 x float> @{{.*}}__spirv_ImageSampleExplicitLod
SYCL_EXTERNAL sycl::float4
test_image_read_sampler(OCLImageTyRead img, const __ocl_sampler_t &smpl) {
  return __invoke__ImageReadSampler<sycl::float4>(img, sycl::float2(0.f, 0.f),
                                                  smpl);
}

// Test ImageArrayRead - __invoke__ImageArrayRead must be inlined,
// __spirv_ImageArrayRead must appear directly in the caller.
// CHECK-LABEL: define {{.*}} @_Z{{.*}}test_image_array_read
// CHECK-NOT: call {{.*}} @{{.*}}__invoke__ImageArrayRead
// CHECK: call {{.*}} <4 x float> @{{.*}}__spirv_ImageArrayRead
SYCL_EXTERNAL sycl::float4 test_image_array_read(OCLImageTyRead img) {
  return __invoke__ImageArrayRead<sycl::float4>(img, sycl::int2(0, 0), 0);
}

// Test ImageArrayWrite - __invoke__ImageArrayWrite must be inlined,
// __spirv_ImageArrayWrite must appear directly in the caller.
// CHECK-LABEL: define {{.*}} @_Z{{.*}}test_image_array_write
// CHECK-NOT: call {{.*}} @{{.*}}__invoke__ImageArrayWrite
// CHECK: call {{.*}} void @{{.*}}__spirv_ImageArrayWrite
SYCL_EXTERNAL void test_image_array_write(OCLImageTyWrite img,
                                          sycl::float4 val) {
  __invoke__ImageArrayWrite(img, sycl::int2(0, 0), 0, val);
}

// Test ImageArrayFetch - __invoke__ImageArrayFetch must be inlined,
// __spirv_ImageArrayFetch must appear directly in the caller.
// CHECK-LABEL: define {{.*}} @_Z{{.*}}test_image_array_fetch
// CHECK-NOT: call {{.*}} @{{.*}}__invoke__ImageArrayFetch
// CHECK: call {{.*}} <4 x float> @{{.*}}__spirv_ImageArrayFetch
SYCL_EXTERNAL sycl::float4 test_image_array_fetch(OCLImageTyRead img) {
  return __invoke__ImageArrayFetch<sycl::float4>(img, sycl::int2(0, 0), 0);
}

// Test SampledImageArrayFetch - __invoke__SampledImageArrayFetch must be
// inlined, __spirv_SampledImageArrayFetch must appear directly in the caller.
// CHECK-LABEL: define {{.*}} @_Z{{.*}}test_sampled_image_array_fetch
// CHECK-NOT: call {{.*}} @{{.*}}__invoke__SampledImageArrayFetch
// CHECK: call {{.*}} <4 x float> @{{.*}}__spirv_SampledImageArrayFetch
SYCL_EXTERNAL sycl::float4
test_sampled_image_array_fetch(OCLSampledImageTy img) {
  return __invoke__SampledImageArrayFetch<sycl::float4>(img, sycl::int2(0, 0),
                                                        0);
}

// Test SampledImageGather - __invoke__SampledImageGather must be inlined,
// __spirv_SampledImageGather must appear directly in the caller.
// CHECK-LABEL: define {{.*}} @_Z{{.*}}test_sampled_image_gather
// CHECK-NOT: call {{.*}} @{{.*}}__invoke__SampledImageGather
// CHECK: call {{.*}} <4 x float> @{{.*}}__spirv_SampledImageGather
SYCL_EXTERNAL sycl::vec<float, 4>
test_sampled_image_gather(OCLSampledImageTy img) {
  return __invoke__SampledImageGather<sycl::vec<float, 4>>(
      img, sycl::float2(0.f, 0.f), 0);
}
