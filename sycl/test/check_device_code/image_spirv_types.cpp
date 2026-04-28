// Check that image operations use supported types by OpenCL SPIR-V env spec.
//
// The OpenCL SPIR-V environment spec requires that OpImageRead, OpImageWrite,
// OpImageFetch, and OpImageSampleExplicitLod use vec4 operands with 32-bit
// component types (or 16-bit for half). This test verifies that channel sizes
// and narrow integer types (int8_t, uint8_t, int16_t, uint16_t) are properly
// widened to vec4 32-bit in the generated SPIR-V calls.

// RUN: %clangxx -O2 -fsycl -fsycl-device-only -fno-discard-value-names -S -emit-llvm -fno-sycl-instrument-device-code -o - %s | FileCheck %s

#include <sycl/sycl.hpp>

using OCLImageTyRead =
    typename sycl::detail::opencl_image_type<2, sycl::access::mode::read,
                                             sycl::access::target::image>::type;

using OCLImageTyWrite =
    typename sycl::detail::opencl_image_type<2, sycl::access::mode::write,
                                             sycl::access::target::image>::type;

using OCLSampledImageTy =
    typename sycl::detail::sampled_opencl_image_type<OCLImageTyRead>::type;

// Test int8_t read - should widen to <4 x i32>
// CHECK-LABEL: define {{.*}} @_Z{{.*}}test_int8_read
// CHECK: call {{.*}} <4 x i32> @{{.*}}__spirv_ImageRead
// CHECK-NOT: call {{.*}} i8 @{{.*}}__spirv_ImageRead
SYCL_EXTERNAL int8_t test_int8_read(OCLImageTyRead img) {
  return __invoke__ImageRead<int8_t>(img, sycl::int2(0, 0));
}

// Test uint8_t read - should widen to <4 x i32>
// CHECK-LABEL: define {{.*}} @_Z{{.*}}test_uint8_read
// CHECK: call {{.*}} <4 x i32> @{{.*}}__spirv_ImageRead
// CHECK-NOT: call {{.*}} i8 @{{.*}}__spirv_ImageRead
SYCL_EXTERNAL uint8_t test_uint8_read(OCLImageTyRead img) {
  return __invoke__ImageRead<uint8_t>(img, sycl::int2(0, 0));
}

// Test int16_t read - should widen to <4 x i32>
// CHECK-LABEL: define {{.*}} @_Z{{.*}}test_int16_read
// CHECK: call {{.*}} <4 x i32> @{{.*}}__spirv_ImageRead
// CHECK-NOT: call {{.*}} i16 @{{.*}}__spirv_ImageRead
SYCL_EXTERNAL int16_t test_int16_read(OCLImageTyRead img) {
  return __invoke__ImageRead<int16_t>(img, sycl::int2(0, 0));
}

// Test uint16_t read - should widen to <4 x i32>
// CHECK-LABEL: define {{.*}} @_Z{{.*}}test_uint16_read
// CHECK: call {{.*}} <4 x i32> @{{.*}}__spirv_ImageRead
// CHECK-NOT: call {{.*}} i16 @{{.*}}__spirv_ImageRead
SYCL_EXTERNAL uint16_t test_uint16_read(OCLImageTyRead img) {
  return __invoke__ImageRead<uint16_t>(img, sycl::int2(0, 0));
}

// Test sycl::vec<int8_t, 4> read - should widen to <4 x i32>
// CHECK-LABEL: define {{.*}} @_Z{{.*}}test_int8_vec4_read
// CHECK: call {{.*}} <4 x i32> @{{.*}}__spirv_ImageRead
// CHECK-NOT: call {{.*}} <4 x i8> @{{.*}}__spirv_ImageRead
SYCL_EXTERNAL sycl::vec<int8_t, 4> test_int8_vec4_read(OCLImageTyRead img) {
  return __invoke__ImageRead<sycl::vec<int8_t, 4>>(img, sycl::int2(0, 0));
}

// Test float - should NOT widen (already 32-bit)
// CHECK-LABEL: define {{.*}} @_Z{{.*}}test_float_read
// CHECK: call {{.*}} <4 x float> @{{.*}}__spirv_ImageRead
// CHECK-NOT: call {{.*}} float @{{.*}}__spirv_ImageRead
SYCL_EXTERNAL float test_float_read(OCLImageTyRead img) {
  return __invoke__ImageRead<float>(img, sycl::int2(0, 0));
}

// Test int32_t - should NOT widen (already 32-bit)
// CHECK-LABEL: define {{.*}} @_Z{{.*}}test_int32_read
// CHECK: call {{.*}} <4 x i32> @{{.*}}__spirv_ImageRead
// CHECK-NOT: call {{.*}} i32 @{{.*}}__spirv_ImageRead
SYCL_EXTERNAL int32_t test_int32_read(OCLImageTyRead img) {
  return __invoke__ImageRead<int32_t>(img, sycl::int2(0, 0));
}

// Test sycl::half - should NOT widen (16-bit is allowed for half)
// CHECK-LABEL: define {{.*}} @_Z{{.*}}test_half_read
// CHECK: call {{.*}} <4 x half> @{{.*}}__spirv_ImageRead
// CHECK-NOT: call {{.*}} half @{{.*}}__spirv_ImageRead
SYCL_EXTERNAL sycl::half test_half_read(OCLImageTyRead img) {
  return __invoke__ImageRead<sycl::half>(img, sycl::int2(0, 0));
}

// Test int8_t write - should widen to <4 x i32>
// CHECK-LABEL: define {{.*}} @_Z{{.*}}test_int8_write
// CHECK: call spir_func void @{{.*}}__spirv_ImageWrite
// CHECK-SAME: <4 x i32>
// CHECK-NOT: call spir_func void @{{.*}}__spirv_ImageWrite{{.*}} i8
SYCL_EXTERNAL void test_int8_write(OCLImageTyWrite img, int8_t val) {
  __invoke__ImageWrite(img, sycl::int2(0, 0), val);
}

// Test uint16_t write - should widen to <4 x i32>
// CHECK-LABEL: define {{.*}} @_Z{{.*}}test_uint16_write
// CHECK: call spir_func void @{{.*}}__spirv_ImageWrite
// CHECK-SAME: <4 x i32>
// CHECK-NOT: call spir_func void @{{.*}}__spirv_ImageWrite{{.*}} i16
SYCL_EXTERNAL void test_uint16_write(OCLImageTyWrite img, uint16_t val) {
  __invoke__ImageWrite(img, sycl::int2(0, 0), val);
}

// Test sycl::vec<uint8_t, 2> write - should widen to <4 x i32> with zero-fill
// CHECK-LABEL: define {{.*}} @_Z{{.*}}test_uint8_vec2_write
// CHECK: call spir_func void @{{.*}}__spirv_ImageWrite
// CHECK-SAME: <4 x i32>
// CHECK-NOT: call spir_func void @{{.*}}__spirv_ImageWrite{{.*}} <2 x i8>
SYCL_EXTERNAL void test_uint8_vec2_write(OCLImageTyWrite img, sycl::vec<uint8_t, 2> val) {
  __invoke__ImageWrite(img, sycl::int2(0, 0), val);
}

// Test int8_t fetch - should widen to <4 x i32>
// CHECK-LABEL: define {{.*}} @_Z{{.*}}test_int8_fetch
// CHECK: call {{.*}} <4 x i32> @{{.*}}__spirv_ImageFetch
// CHECK-NOT: call {{.*}} i8 @{{.*}}__spirv_ImageFetch
SYCL_EXTERNAL int8_t test_int8_fetch(OCLImageTyRead img) {
  return __invoke__ImageFetch<int8_t>(img, sycl::int2(0, 0));
}

// Test uint16_t fetch - should widen to <4 x i32>
// CHECK-LABEL: define {{.*}} @_Z{{.*}}test_uint16_fetch
// CHECK: call {{.*}} <4 x i32> @{{.*}}__spirv_ImageFetch
// CHECK-NOT: call {{.*}} i16 @{{.*}}__spirv_ImageFetch
SYCL_EXTERNAL uint16_t test_uint16_fetch(OCLImageTyRead img) {
  return __invoke__ImageFetch<uint16_t>(img, sycl::int2(0, 0));
}

// Test int8_t sampled fetch - should widen to <4 x i32>
// CHECK-LABEL: define {{.*}} @_Z{{.*}}test_int8_sampled_fetch
// CHECK: call {{.*}} <4 x i32> @{{.*}}__spirv_SampledImageFetch
// CHECK-NOT: call {{.*}} i8 @{{.*}}__spirv_SampledImageFetch
SYCL_EXTERNAL int8_t test_int8_sampled_fetch(OCLSampledImageTy img) {
  return __invoke__SampledImageFetch<int8_t>(img, sycl::int2(0, 0));
}

// Test uint16_t sampled fetch - should widen to <4 x i32>
// CHECK-LABEL: define {{.*}} @_Z{{.*}}test_uint16_sampled_fetch
// CHECK: call {{.*}} <4 x i32> @{{.*}}__spirv_SampledImageFetch
// CHECK-NOT: call {{.*}} i16 @{{.*}}__spirv_SampledImageFetch
SYCL_EXTERNAL uint16_t test_uint16_sampled_fetch(OCLSampledImageTy img) {
  return __invoke__SampledImageFetch<uint16_t>(img, sycl::int2(0, 0));
}

// Test int8_t array read - should widen to <4 x i32>
// CHECK-LABEL: define {{.*}} @_Z{{.*}}test_int8_array_read
// CHECK: call {{.*}} <4 x i32> @{{.*}}__spirv_ImageArrayRead
// CHECK-NOT: call {{.*}} i8 @{{.*}}__spirv_ImageArrayRead
SYCL_EXTERNAL int8_t test_int8_array_read(OCLImageTyRead img) {
  return __invoke__ImageArrayRead<int8_t>(img, sycl::int2(0, 0), 0);
}

// Test uint16_t array read - should widen to <4 x i32>
// CHECK-LABEL: define {{.*}} @_Z{{.*}}test_uint16_array_read
// CHECK: call {{.*}} <4 x i32> @{{.*}}__spirv_ImageArrayRead
// CHECK-NOT: call {{.*}} i16 @{{.*}}__spirv_ImageArrayRead
SYCL_EXTERNAL uint16_t test_uint16_array_read(OCLImageTyRead img) {
  return __invoke__ImageArrayRead<uint16_t>(img, sycl::int2(0, 0), 0);
}

// Test int8_t array write - should widen to <4 x i32>
// CHECK-LABEL: define {{.*}} @_Z{{.*}}test_int8_array_write
// CHECK: call spir_func void @{{.*}}__spirv_ImageArrayWrite
// CHECK-SAME: <4 x i32>
// CHECK-NOT: call spir_func void @{{.*}}__spirv_ImageArrayWrite{{.*}} i8
SYCL_EXTERNAL void test_int8_array_write(OCLImageTyWrite img, int8_t val) {
  __invoke__ImageArrayWrite(img, sycl::int2(0, 0), 0, val);
}

// Test uint16_t array write - should widen to <4 x i32>
// CHECK-LABEL: define {{.*}} @_Z{{.*}}test_uint16_array_write
// CHECK: call spir_func void @{{.*}}__spirv_ImageArrayWrite
// CHECK-SAME: <4 x i32>
// CHECK-NOT: call spir_func void @{{.*}}__spirv_ImageArrayWrite{{.*}} i16
SYCL_EXTERNAL void test_uint16_array_write(OCLImageTyWrite img, uint16_t val) {
  __invoke__ImageArrayWrite(img, sycl::int2(0, 0), 0, val);
}

// Test int8_t array fetch - should widen to <4 x i32>
// CHECK-LABEL: define {{.*}} @_Z{{.*}}test_int8_array_fetch
// CHECK: call {{.*}} <4 x i32> @{{.*}}__spirv_ImageArrayFetch
// CHECK-NOT: call {{.*}} i8 @{{.*}}__spirv_ImageArrayFetch
SYCL_EXTERNAL int8_t test_int8_array_fetch(OCLImageTyRead img) {
  return __invoke__ImageArrayFetch<int8_t>(img, sycl::int2(0, 0), 0);
}

// Test uint16_t array fetch - should widen to <4 x i32>
// CHECK-LABEL: define {{.*}} @_Z{{.*}}test_uint16_array_fetch
// CHECK: call {{.*}} <4 x i32> @{{.*}}__spirv_ImageArrayFetch
// CHECK-NOT: call {{.*}} i16 @{{.*}}__spirv_ImageArrayFetch
SYCL_EXTERNAL uint16_t test_uint16_array_fetch(OCLImageTyRead img) {
  return __invoke__ImageArrayFetch<uint16_t>(img, sycl::int2(0, 0), 0);
}

// Test int8_t sampled array fetch - should widen to <4 x i32>
// CHECK-LABEL: define {{.*}} @_Z{{.*}}test_int8_sampled_array_fetch
// CHECK: call {{.*}} <4 x i32> @{{.*}}__spirv_SampledImageArrayFetch
// CHECK-NOT: call {{.*}} i8 @{{.*}}__spirv_SampledImageArrayFetch
SYCL_EXTERNAL int8_t test_int8_sampled_array_fetch(OCLSampledImageTy img) {
  return __invoke__SampledImageArrayFetch<int8_t>(img, sycl::int2(0, 0), 0);
}

// Test uint16_t sampled array fetch - should widen to <4 x i32>
// CHECK-LABEL: define {{.*}} @_Z{{.*}}test_uint16_sampled_array_fetch
// CHECK: call {{.*}} <4 x i32> @{{.*}}__spirv_SampledImageArrayFetch
// CHECK-NOT: call {{.*}} i16 @{{.*}}__spirv_SampledImageArrayFetch
SYCL_EXTERNAL uint16_t test_uint16_sampled_array_fetch(OCLSampledImageTy img) {
  return __invoke__SampledImageArrayFetch<uint16_t>(img, sycl::int2(0, 0), 0);
}

// Test int8_t cubemap read - should widen to <4 x i32>
// CHECK-LABEL: define {{.*}} @_Z{{.*}}test_int8_read_cubemap
// CHECK: call {{.*}} <4 x i32> @{{.*}}__spirv_ImageSampleCubemap
// CHECK-NOT: call {{.*}} i8 @{{.*}}__spirv_ImageSampleCubemap
SYCL_EXTERNAL int8_t test_int8_read_cubemap(OCLSampledImageTy img) {
  return __invoke__ImageReadCubemap<int8_t>(img, sycl::float3(0.0f, 0.0f, 0.0f));
}

// Test uint16_t cubemap read - should widen to <4 x i32>
// CHECK-LABEL: define {{.*}} @_Z{{.*}}test_uint16_read_cubemap
// CHECK: call {{.*}} <4 x i32> @{{.*}}__spirv_ImageSampleCubemap
// CHECK-NOT: call {{.*}} i16 @{{.*}}__spirv_ImageSampleCubemap
SYCL_EXTERNAL uint16_t test_uint16_read_cubemap(OCLSampledImageTy img) {
  return __invoke__ImageReadCubemap<uint16_t>(img, sycl::float3(0.0f, 0.0f, 0.0f));
}

// Test int8_t lod read - should widen to <4 x i32>
// CHECK-LABEL: define {{.*}} @_Z{{.*}}test_int8_read_lod
// CHECK: call {{.*}} <4 x i32> @{{.*}}__spirv_ImageSampleExplicitLod
// CHECK-NOT: call {{.*}} i8 @{{.*}}__spirv_ImageSampleExplicitLod
SYCL_EXTERNAL int8_t test_int8_read_lod(OCLSampledImageTy img) {
  return __invoke__ImageReadLod<int8_t>(img, sycl::float2(0.0f, 0.0f), 0.0f);
}

// Test uint16_t lod read - should widen to <4 x i32>
// CHECK-LABEL: define {{.*}} @_Z{{.*}}test_uint16_read_lod
// CHECK: call {{.*}} <4 x i32> @{{.*}}__spirv_ImageSampleExplicitLod
// CHECK-NOT: call {{.*}} i16 @{{.*}}__spirv_ImageSampleExplicitLod
SYCL_EXTERNAL uint16_t test_uint16_read_lod(OCLSampledImageTy img) {
  return __invoke__ImageReadLod<uint16_t>(img, sycl::float2(0.0f, 0.0f), 0.0f);
}

// Test int8_t grad read - should widen to <4 x i32>
// CHECK-LABEL: define {{.*}} @_Z{{.*}}test_int8_read_grad
// CHECK: call {{.*}} <4 x i32> @{{.*}}__spirv_ImageSampleExplicitLod
// CHECK-NOT: call {{.*}} i8 @{{.*}}__spirv_ImageSampleExplicitLod
SYCL_EXTERNAL int8_t test_int8_read_grad(OCLSampledImageTy img) {
  return __invoke__ImageReadGrad<int8_t>(img, sycl::float2(0.0f, 0.0f),
                                         sycl::float2(0.0f, 0.0f),
                                         sycl::float2(0.0f, 0.0f));
}

// Test uint16_t grad read - should widen to <4 x i32>
// CHECK-LABEL: define {{.*}} @_Z{{.*}}test_uint16_read_grad
// CHECK: call {{.*}} <4 x i32> @{{.*}}__spirv_ImageSampleExplicitLod
// CHECK-NOT: call {{.*}} i16 @{{.*}}__spirv_ImageSampleExplicitLod
SYCL_EXTERNAL uint16_t test_uint16_read_grad(OCLSampledImageTy img) {
  return __invoke__ImageReadGrad<uint16_t>(img, sycl::float2(0.0f, 0.0f),
                                            sycl::float2(0.0f, 0.0f),
                                            sycl::float2(0.0f, 0.0f));
}

// Test int8_t sampler read - should widen to <4 x i32>
// CHECK-LABEL: define {{.*}} @_Z{{.*}}test_int8_read_sampler
// CHECK: call {{.*}} <4 x i32> @{{.*}}__spirv_ImageSampleExplicitLod
// CHECK-NOT: call {{.*}} i8 @{{.*}}__spirv_ImageSampleExplicitLod
SYCL_EXTERNAL int8_t test_int8_read_sampler(OCLImageTyRead img,
                                             const __ocl_sampler_t &smpl) {
  return __invoke__ImageReadSampler<int8_t>(img, sycl::float2(0.0f, 0.0f), smpl);
}

// Test uint16_t sampler read - should widen to <4 x i32>
// CHECK-LABEL: define {{.*}} @_Z{{.*}}test_uint16_read_sampler
// CHECK: call {{.*}} <4 x i32> @{{.*}}__spirv_ImageSampleExplicitLod
// CHECK-NOT: call {{.*}} i16 @{{.*}}__spirv_ImageSampleExplicitLod
SYCL_EXTERNAL uint16_t test_uint16_read_sampler(OCLImageTyRead img,
                                                 const __ocl_sampler_t &smpl) {
  return __invoke__ImageReadSampler<uint16_t>(img, sycl::float2(0.0f, 0.0f), smpl);
}
