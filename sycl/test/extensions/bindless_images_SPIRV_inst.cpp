// Check SPV_INTEL_bindless_images instructions are emitted correctly.

// RUN: %clangxx -fsycl -fsycl-device-only -fsycl-targets=spir64-unknown-unknown %s -o %t.out
// RUN: llvm-spirv -spirv-ext=+SPV_INTEL_bindless_images %t.out -spirv-text -o %t.out.spv
// RUN: FileCheck %s --input-file %t.out.spv

#include <iostream>
#include <sycl/sycl.hpp>

// Data type of image, sampler and sampled image handles
// Arguments: Result, width, signedness
// CHECK: TypeInt 2 64 0

// Create dummy image and sampled image handles
// 2 here, represents a 64-bit int type
// Arguments: Result Type, Result, Literal Value
// CHECK: Constant 2 [[IMAGEID:[0-9]+]] 123
// CHECK: Constant 2 [[IMAGETWOID:[0-9]+]] 1234
// CHECK: Constant 2 [[SAMPLERID:[0-9]+]] 12345
// CHECK: Constant 2 [[SAMPLEDIMAGEID:[0-9]+]] 123456

// Generate the appropriate `Result Type`s used by `ConvertHandleToImageINTEL`,
// `ConvertHandleToSamplerINTEL` and `ConvertHandleToSampledImageINTEL`. Operand
// `7` here represents 'TypeVoid`. Must be `TypeVoid` as the type of the image
// is not known at compile time. The last operand is the access qualifier. With
// 0 read only and 1 write only.
// Arguments: TypeImage, Result, Sampled Type, Dim, Depth, Arrayed, MS, Sampled
// and Image Format.
// CHECK: TypeImage [[IMAGETYPEREAD:[0-9]+]] 7 0 0 0 0 0 0 0
// CHECK: TypeImage [[IMAGETYPEWRITE:[0-9]+]] 7 0 0 0 0 0 0 1
// Generate `Result Type` for samplers
// Arguments: Result
// CHECK: TypeSampler [[SAMPLERTYPE:[0-9]+]]
// Generate `Result Type` for sampled images
// Arguments: Result, Image Type
// CHECK: TypeSampledImage [[SAMPLEDIMAGETYPE:[0-9]+]] [[IMAGETYPEREAD]]

// Convert handles to SPIR-V images, samplers and sampled images
// Arguments: Result Type, Result, Handle
// CHECK: ConvertHandleToImageINTEL [[IMAGETYPEREAD]] {{[0-9]+}} [[IMAGEID]]
// CHECK: ConvertHandleToImageINTEL [[IMAGETYPEWRITE]] {{[0-9]+}} [[IMAGETWOID]]
// CHECK: ConvertHandleToSamplerINTEL [[SAMPLERTYPE]] {{[0-9]+}} [[SAMPLERID]]
// CHECK: ConvertHandleToSampledImageINTEL [[SAMPLEDIMAGETYPE]] {{[0-9]+}} [[SAMPLEDIMAGEID]]

#include <sycl/detail/image_ocl_types.hpp>

#ifdef __SYCL_DEVICE_ONLY__
template <int NDims>
using OCLImageTyRead =
    typename sycl::detail::opencl_image_type<NDims, sycl::access::mode::read,
                                             sycl::access::target::image>::type;

template <int NDims>
using OCLImageTyWrite =
    typename sycl::detail::opencl_image_type<NDims, sycl::access::mode::write,
                                             sycl::access::target::image>::type;

template <int NDims>
using OCLSampledImageTy = typename sycl::detail::sampled_opencl_image_type<
    OCLImageTyRead<NDims>>::type;
#endif

template <typename ReturnT> ReturnT handleToImage(const uint64_t &imageHandle) {
#ifdef __SYCL_DEVICE_ONLY__
#if defined(__SPIR__)
  return __spirv_ConvertHandleToImageINTEL<ReturnT>(imageHandle);
#endif
#else
  assert(false); // Bindless images not yet implemented on host.
#endif
}

template <typename ReturnT>
ReturnT handleToSampler(const uint64_t &samplerHandle) {
#ifdef __SYCL_DEVICE_ONLY__
#if defined(__SPIR__)
  return __spirv_ConvertHandleToSamplerINTEL<ReturnT>(samplerHandle);
#endif
#else
  assert(false); // Bindless images not yet implemented on host.
#endif
}

template <typename ReturnT>
ReturnT handleToSampledImage(const uint64_t &sampledImageHandle) {
#ifdef __SYCL_DEVICE_ONLY__
#if defined(__SPIR__)
  return __spirv_ConvertHandleToSampledImageINTEL<ReturnT>(sampledImageHandle);
#endif
#else
  assert(false); // Bindless images not yet implemented on host.
#endif
}

class image_addition;

int main() {

  sycl::device dev;
  sycl::queue q(dev);
  auto ctxt = q.get_context();
  try {
    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<image_addition>(1, [=](sycl::id<1> id) {
#ifdef __SYCL_DEVICE_ONLY__
        OCLImageTyRead<1> imageRead = handleToImage<OCLImageTyRead<1>>(123);

        OCLImageTyWrite<1> imageWrite = handleToImage<OCLImageTyWrite<1>>(1234);

        __ocl_sampler_t sampler = handleToSampler<__ocl_sampler_t>(12345);

        OCLSampledImageTy<1> sampImage =
            handleToSampledImage<OCLSampledImageTy<1>>(123456);
#endif
      });
    });

  } catch (sycl::exception e) {
    std::cerr << "SYCL exception caught! : " << e.what() << "\n";
    return 1;
  } catch (...) {
    std::cerr << "Unknown exception caught!\n";
    return 2;
  }
}
