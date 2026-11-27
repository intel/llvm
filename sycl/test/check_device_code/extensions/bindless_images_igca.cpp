// RUN: %clangxx -O2 -fsycl -fsycl-device-only -S -emit-llvm -fno-sycl-instrument-device-code -o - %s | FileCheck %s --check-prefix=IMAGES
// RUN: %clangxx -O2 -fsycl -fsycl-device-only -S -emit-llvm -fno-sycl-instrument-device-code -D__OFFLOAD_ARCH_INTEL_GENERIC__=1 -o - %s | FileCheck %s --check-prefix=IMAGES
// RUN: %clangxx -O2 -fsycl -fsycl-device-only -S -emit-llvm -fno-sycl-instrument-device-code -D__OFFLOAD_ARCH_IGCA__=30 -D__OFFLOAD_ARCH_IGCA_RENDER__=30 -o - %s | FileCheck %s --check-prefix=IMAGES
// RUN: %clangxx -O2 -fsycl -fsycl-device-only -S -emit-llvm -fno-sycl-instrument-device-code -D__OFFLOAD_ARCH_IGCA__=20 -o - %s | FileCheck %s --check-prefix=NO-IMAGES --implicit-check-not=spirv.Image --implicit-check-not=spirv.SampledImage --implicit-check-not=__spirv_Image --implicit-check-not=__spirv_SampledImage --implicit-check-not=__spirv_ConvertHandleTo
// RUN: %clangxx -O2 -fsycl -fsycl-device-only -S -emit-llvm -fno-sycl-instrument-device-code -D__OFFLOAD_ARCH_IGCA__=30 -o - %s | FileCheck %s --check-prefix=NO-IMAGES --implicit-check-not=spirv.Image --implicit-check-not=spirv.SampledImage --implicit-check-not=__spirv_Image --implicit-check-not=__spirv_SampledImage --implicit-check-not=__spirv_ConvertHandleTo

// Images are only supported by IGCA targets with the render feature set. For
// other IGCA targets the compiler must not emit instructions that the specified
// IGCA target does not support.

#include <sycl/sycl.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

// IMAGES-LABEL: define {{.*}}spir_kernel {{.*}}FetchUnsampled(
// IMAGES: call {{.*}}__spirv_ConvertHandleToImageINTEL
// IMAGES: call {{.*}}__spirv_ImageRead
// NO-IMAGES-LABEL: define {{.*}}spir_kernel {{.*}}FetchUnsampled(
// NO-IMAGES: call {{.*}}@__assert_fail

// IMAGES-LABEL: define {{.*}}spir_kernel {{.*}}Fetch1D(
// IMAGES: call {{.*}}__spirv_ConvertHandleToImageINTEL
// IMAGES: call {{.*}}__spirv_ImageRead
// NO-IMAGES-LABEL: define {{.*}}spir_kernel {{.*}}Fetch1D(
// NO-IMAGES: call {{.*}}@__assert_fail

// IMAGES-LABEL: define {{.*}}spir_kernel {{.*}}FetchSampled(
// IMAGES: call {{.*}}__spirv_ConvertHandleToImageINTEL
// IMAGES: call {{.*}}__spirv_ImageRead
// NO-IMAGES-LABEL: define {{.*}}spir_kernel {{.*}}FetchSampled(
// NO-IMAGES: call {{.*}}@__assert_fail

// IMAGES-LABEL: define {{.*}}spir_kernel {{.*}}Write(
// IMAGES: call {{.*}}__spirv_ConvertHandleToImageINTEL
// IMAGES: call {{.*}}__spirv_ImageWrite
// NO-IMAGES-LABEL: define {{.*}}spir_kernel {{.*}}Write(
// NO-IMAGES: call {{.*}}@__assert_fail

// IMAGES-LABEL: define {{.*}}spir_kernel {{.*}}Sample(
// IMAGES: call {{.*}}__spirv_ConvertHandleToSampledImageINTEL
// IMAGES: call {{.*}}__spirv_ImageSampleExplicitLod
// NO-IMAGES-LABEL: define {{.*}}spir_kernel {{.*}}Sample(
// NO-IMAGES: call {{.*}}@__assert_fail

// IMAGES-LABEL: define {{.*}}spir_kernel {{.*}}SampleMipmapLevel(
// IMAGES: call {{.*}}__spirv_ConvertHandleToSampledImageINTEL
// IMAGES: call {{.*}}__spirv_ImageSampleExplicitLod
// NO-IMAGES-LABEL: define {{.*}}spir_kernel {{.*}}SampleMipmapLevel(
// NO-IMAGES: call {{.*}}@__assert_fail

// IMAGES-LABEL: define {{.*}}spir_kernel {{.*}}SampleMipmapGrad(
// IMAGES: call {{.*}}__spirv_ConvertHandleToSampledImageINTEL
// IMAGES: call {{.*}}__spirv_ImageSampleExplicitLod
// NO-IMAGES-LABEL: define {{.*}}spir_kernel {{.*}}SampleMipmapGrad(
// NO-IMAGES: call {{.*}}@__assert_fail

// IMAGES-LABEL: define {{.*}}spir_kernel {{.*}}FetchArrayUnsampled(
// IMAGES: call {{.*}}__spirv_ConvertHandleToImageINTEL
// IMAGES: call {{.*}}__spirv_ImageRead
// NO-IMAGES-LABEL: define {{.*}}spir_kernel {{.*}}FetchArrayUnsampled(
// NO-IMAGES: call {{.*}}@__assert_fail

// IMAGES-LABEL: define {{.*}}spir_kernel {{.*}}FetchArraySampled(
// IMAGES: call {{.*}}__spirv_ConvertHandleToSampledImageINTEL
// IMAGES: call {{.*}}__spirv_ImageSampleExplicitLod
// NO-IMAGES-LABEL: define {{.*}}spir_kernel {{.*}}FetchArraySampled(
// NO-IMAGES: call {{.*}}@__assert_fail

// IMAGES-LABEL: define {{.*}}spir_kernel {{.*}}SampleArray(
// IMAGES: call {{.*}}__spirv_ConvertHandleToSampledImageINTEL
// IMAGES: call {{.*}}__spirv_ImageSampleExplicitLod
// NO-IMAGES-LABEL: define {{.*}}spir_kernel {{.*}}SampleArray(
// NO-IMAGES: call {{.*}}@__assert_fail

// IMAGES-LABEL: define {{.*}}spir_kernel {{.*}}WriteArray(
// IMAGES: call {{.*}}__spirv_ConvertHandleToImageINTEL
// IMAGES: call {{.*}}__spirv_ImageWrite
// NO-IMAGES-LABEL: define {{.*}}spir_kernel {{.*}}WriteArray(
// NO-IMAGES: call {{.*}}@__assert_fail

// sample_cubemap does not compile for SPIR-V on targets with images, so it is
// only checked on targets without images.
// NO-IMAGES-LABEL: define {{.*}}spir_kernel {{.*}}SampleCubemap(
// NO-IMAGES: call {{.*}}@__assert_fail

int main() {
  sycl::queue Q;
  sycl::int4 *Out = sycl::malloc_shared<sycl::int4>(1, Q);
  sycl::float4 *FOut = sycl::malloc_shared<sycl::float4>(1, Q);
  syclexp::unsampled_image_handle UH{0};
  syclexp::sampled_image_handle SH{0};

  Q.single_task<class FetchUnsampled>(
      [=] { Out[0] = syclexp::fetch_image<sycl::int4>(UH, sycl::int2{0, 0}); });
  Q.single_task<class Fetch1D>(
      [=] { Out[0] = syclexp::fetch_image<sycl::int4>(UH, 0); });
  Q.single_task<class FetchSampled>([=] {
    FOut[0] = syclexp::fetch_image<sycl::float4>(SH, sycl::int2{0, 0});
  });
  Q.single_task<class Write>(
      [=] { syclexp::write_image(UH, sycl::int2{0, 0}, sycl::int4{1}); });
  Q.single_task<class Sample>([=] {
    FOut[0] = syclexp::sample_image<sycl::float4>(SH, sycl::float2{0, 0});
  });
  Q.single_task<class SampleMipmapLevel>([=] {
    FOut[0] = syclexp::sample_mipmap<sycl::float4>(SH, sycl::float2{0, 0}, 1.f);
  });
  Q.single_task<class SampleMipmapGrad>([=] {
    FOut[0] = syclexp::sample_mipmap<sycl::float4>(
        SH, sycl::float2{0, 0}, sycl::float2{1, 0}, sycl::float2{0, 1});
  });
  Q.single_task<class FetchArrayUnsampled>([=] {
    Out[0] = syclexp::fetch_image_array<sycl::int4>(UH, sycl::int2{0, 0}, 1);
  });
  Q.single_task<class FetchArraySampled>([=] {
    FOut[0] = syclexp::fetch_image_array<sycl::float4>(SH, sycl::int2{0, 0}, 1);
  });
  Q.single_task<class SampleArray>([=] {
    FOut[0] =
        syclexp::sample_image_array<sycl::float4>(SH, sycl::float2{0, 0}, 1);
  });
  Q.single_task<class WriteArray>([=] {
    syclexp::write_image_array(UH, sycl::int2{0, 0}, 1, sycl::int4{1});
  });

#if defined(__OFFLOAD_ARCH_IGCA__) && !defined(__OFFLOAD_ARCH_IGCA_RENDER__)
  Q.single_task<class SampleCubemap>([=] {
    FOut[0] = syclexp::sample_cubemap<sycl::float4>(SH, sycl::float3{0, 0, 1});
  });
#endif
}
