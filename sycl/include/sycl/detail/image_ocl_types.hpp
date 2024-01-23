//===-- Image_ocl_types.hpp - Image OpenCL types --------- ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file is to define some utility functions and declare the structs with
// type as appropriate opencl image types based on Dims, AccessMode and
// AccessTarget. The macros essentially expand to -
//
// template <> struct
// opencl_image_type<1, access::mode::read, access::target::image> {
//   using type = __ocl_image1d_ro_t;
// };
//
// template <>
// struct opencl_image_type<1, access::mode::write, access::target::image> {
//   using type = __ocl_image1d_array_wo_t;
// };
//
// As an example, this can be
// used as below:
// detail::opencl_image_type<2, access::mode::read, access::target::image>::type
//    MyImage;
//
#pragma once

#ifdef __SYCL_DEVICE_ONLY__

#include <sycl/access/access.hpp>
#include <sycl/detail/generic_type_traits.hpp>

#include <CL/__spirv/spirv_ops.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {

// Type trait to get the associated sampled image type for a given image type.
template <typename ImageType> struct sampled_opencl_image_type;

} // namespace detail
} // namespace _V1
} // namespace sycl

#define __SYCL_INVOKE_SPIRV_CALL_ARG1(call)                                    \
  template <typename R, typename T1> inline R __invoke_##call(T1 ParT1) {      \
    using Ret = sycl::detail::ConvertToOpenCLType_t<R>;                        \
    T1 Arg1 = ParT1;                                                           \
    Ret RetVar = __spirv_##call<Ret, T1>(Arg1);                                \
    return sycl::detail::convertDataToType<Ret, R>(RetVar);                    \
  }

// The macro defines the function __invoke_ImageXXXX,
// The functions contains the spirv call to __spirv_ImageXXXX.
__SYCL_INVOKE_SPIRV_CALL_ARG1(ImageQuerySize)
__SYCL_INVOKE_SPIRV_CALL_ARG1(ImageQueryFormat)
__SYCL_INVOKE_SPIRV_CALL_ARG1(ImageQueryOrder)

template <typename ImageT, typename CoordT, typename ValT>
static void __invoke__ImageWrite(ImageT Img, CoordT Coords, ValT Val) {

  // Convert from sycl types to builtin types to get correct function mangling.
  using TmpValT = sycl::detail::ConvertToOpenCLType_t<ValT>;
  using TmpCoordT = sycl::detail::ConvertToOpenCLType_t<CoordT>;

  TmpCoordT TmpCoord =
      sycl::detail::convertDataToType<CoordT, TmpCoordT>(Coords);
  TmpValT TmpVal = sycl::detail::convertDataToType<ValT, TmpValT>(Val);
  __spirv_ImageWrite<ImageT, TmpCoordT, TmpValT>(Img, TmpCoord, TmpVal);
}

template <typename RetType, typename ImageT, typename CoordT>
static RetType __invoke__ImageRead(ImageT Img, CoordT Coords) {

  // Convert from sycl types to builtin types to get correct function mangling.
  using TempRetT = sycl::detail::ConvertToOpenCLType_t<RetType>;
  using TempArgT = sycl::detail::ConvertToOpenCLType_t<CoordT>;

  TempArgT Arg = sycl::detail::convertDataToType<CoordT, TempArgT>(Coords);
  TempRetT Ret = __spirv_ImageRead<TempRetT, ImageT, TempArgT>(Img, Arg);
  return sycl::detail::convertDataToType<TempRetT, RetType>(Ret);
}

template <typename RetType, typename ImageT, typename CoordT>
static RetType __invoke__ImageArrayRead(ImageT Img, CoordT Coords,
                                        int ArrayLayer) {

  // Convert from sycl types to builtin types to get correct function mangling.
  using TempRetT = sycl::detail::ConvertToOpenCLType_t<RetType>;
  using TempArgT = sycl::detail::ConvertToOpenCLType_t<CoordT>;

  TempArgT Arg = sycl::detail::convertDataToType<CoordT, TempArgT>(Coords);
  TempRetT Ret =
      __spirv_ImageArrayRead<TempRetT, ImageT, TempArgT>(Img, Arg, ArrayLayer);
  return sycl::detail::convertDataToType<TempRetT, RetType>(Ret);
}

template <typename ImageT, typename CoordT, typename ValT>
static void __invoke__ImageArrayWrite(ImageT Img, CoordT Coords, int ArrayLayer,
                                      ValT Val) {

  // Convert from sycl types to builtin types to get correct function mangling.
  using TmpValT = sycl::detail::ConvertToOpenCLType_t<ValT>;
  using TmpCoordT = sycl::detail::ConvertToOpenCLType_t<CoordT>;

  TmpCoordT TmpCoord =
      sycl::detail::convertDataToType<CoordT, TmpCoordT>(Coords);
  TmpValT TmpVal = sycl::detail::convertDataToType<ValT, TmpValT>(Val);
  __spirv_ImageArrayWrite<ImageT, TmpCoordT, TmpValT>(Img, TmpCoord, ArrayLayer,
                                                      TmpVal);
}

template <typename RetType, typename SmpImageT, typename CoordT>
static RetType __invoke__ImageReadLod(SmpImageT SmpImg, CoordT Coords,
                                      float Level) {

  // Convert from sycl types to builtin types to get correct function mangling.
  using TempRetT = sycl::detail::ConvertToOpenCLType_t<RetType>;
  using TempArgT = sycl::detail::ConvertToOpenCLType_t<CoordT>;

  TempArgT TmpCoords =
      sycl::detail::convertDataToType<CoordT, TempArgT>(Coords);

  enum ImageOperands { Lod = 0x2 };

  // OpImageSampleExplicitLod
  // Its components must be the same as Sampled Type of the underlying
  // OpTypeImage
  // Sampled Image must be an object whose type is OpTypeSampledImage
  // Image Operands encodes what operands follow. Either Lod
  // or Grad image operands must be present
  TempRetT Ret = __spirv_ImageSampleExplicitLod<SmpImageT, TempRetT, TempArgT>(
      SmpImg, TmpCoords, ImageOperands::Lod, Level);
  return sycl::detail::convertDataToType<TempRetT, RetType>(Ret);
}

template <typename RetType, typename SmpImageT, typename CoordT>
static RetType __invoke__ImageReadGrad(SmpImageT SmpImg, CoordT Coords,
                                       CoordT Dx, CoordT Dy) {

  // Convert from sycl types to builtin types to get correct function mangling.
  using TempRetT = sycl::detail::ConvertToOpenCLType_t<RetType>;
  using TempArgT = sycl::detail::ConvertToOpenCLType_t<CoordT>;

  TempArgT TmpCoords =
      sycl::detail::convertDataToType<CoordT, TempArgT>(Coords);
  TempArgT TmpGraddX = sycl::detail::convertDataToType<CoordT, TempArgT>(Dx);
  TempArgT TmpGraddY = sycl::detail::convertDataToType<CoordT, TempArgT>(Dy);

  enum ImageOperands { Grad = 0x3 };

  // OpImageSampleExplicitLod
  // Its components must be the same as Sampled Type of the underlying
  // OpTypeImage
  // Sampled Image must be an object whose type is OpTypeSampledImage
  // Image Operands encodes what operands follow. Either Lod
  // or Grad image operands must be present
  TempRetT Ret = __spirv_ImageSampleExplicitLod<SmpImageT, TempRetT, TempArgT>(
      SmpImg, TmpCoords, ImageOperands::Grad, TmpGraddX, TmpGraddY);
  return sycl::detail::convertDataToType<TempRetT, RetType>(Ret);
}

template <typename RetType, typename ImageT, typename CoordT>
static RetType __invoke__ImageReadSampler(ImageT Img, CoordT Coords,
                                          const __ocl_sampler_t &Smpl) {

  // Convert from sycl types to builtin types to get correct function mangling.
  using TempRetT = sycl::detail::ConvertToOpenCLType_t<RetType>;
  using TempArgT = sycl::detail::ConvertToOpenCLType_t<CoordT>;
  using SampledT =
      typename sycl::detail::sampled_opencl_image_type<ImageT>::type;

  TempArgT TmpCoords =
      sycl::detail::convertDataToType<CoordT, TempArgT>(Coords);
  // According to validation rules(SPIR-V specification, section 2.16.1) result
  // of __spirv_SampledImage is allowed to be an operand of image lookup
  // and query instructions explicitly specified to take an operand whose
  // type is OpTypeSampledImage.
  //
  // According to SPIR-V specification section 3.32.10 at least one operand
  // setting the level of detail must be present. The last two arguments of
  // __spirv_ImageSampleExplicitLod represent image operand type and value.
  // From the SPIR-V specification section 3.14:
  enum ImageOperands { Lod = 0x2 };

  // Lod value is zero as mipmap is not supported.
  TempRetT Ret = __spirv_ImageSampleExplicitLod<SampledT, TempRetT, TempArgT>(
      __spirv_SampledImage<ImageT, SampledT>(Img, Smpl), TmpCoords,
      ImageOperands::Lod, 0.0f);
  return sycl::detail::convertDataToType<TempRetT, RetType>(Ret);
}

namespace sycl {
inline namespace _V1 {
namespace detail {

// Function to return the number of channels for Image Channel Order returned by
// SPIR-V call to OpImageQueryOrder.
// The returned int value represents an enum from Image Channel Order. The enums
// for Image Channel Order are mapped differently in sycl and SPIR-V spec.
inline int getSPIRVNumChannels(int ImageChannelOrder) {
  switch (ImageChannelOrder) {
  case 0:  // R
  case 1:  // A
  case 10: // Rx
  case 8:  // Intensity
  case 9:  // Luminance
    return 1;
  case 2:  // RG
  case 3:  // RA
  case 11: // RGx
    return 2;
  case 4: // RGB
    return 3;
  case 5:  // RGBA
  case 6:  // BGRA
  case 7:  // ARGB
  case 12: // RGBx
  case 19: // ABGR
  case 17: // sRGBA
    return 4;
  case 13: // Depth
  case 14: // DepthStencil
  case 18: // sBGRA
           // TODO: Enable the below assert after assert is supported for device
           // compiler. assert(!"Unhandled image channel order in sycl.");
  default:
    return 0;
  }
}

// Function to compute the Element Size for a given Image Channel Type and Image
// Channel Order, returned by SPIR-V calls to OpImageQueryFormat and
// OpImageQueryOrder respectively.
// The returned int value from OpImageQueryFormat represents an enum from Image
// Channel Data Type. The enums for Image Channel Data Type are mapped
// differently in sycl and SPIR-V spec.
inline int getSPIRVElementSize(int ImageChannelType, int ImageChannelOrder) {
  int NumChannels = getSPIRVNumChannels(ImageChannelOrder);
  switch (ImageChannelType) {
  case 0:  // SnormInt8
  case 2:  // UnormInt8
  case 7:  // SignedInt8
  case 10: // UnsignedInt8
    return NumChannels;
  case 1:  // SnormInt16
  case 3:  // UnormInt16
  case 8:  // SignedInt16
  case 11: // UnsignedInt16
  case 13: // HalfFloat
    return 2 * NumChannels;
  case 4: // UnormShort565
  case 5: // UnormShort555
    return 2;
  case 6: // UnormInt101010
    return 4;
  case 9:  // SignedInt32
  case 12: // UnsignedInt32
  case 14: // Float
    return 4 * NumChannels;
  case 15: // UnormInt24
  case 16: // UnormInt101010_2
  default:
    // TODO: Enable the below assert after assert is supported for device
    // compiler. assert(!"Unhandled image channel type in sycl.");
    return 0;
  }
}

template <int Dimensions, access::mode AccessMode, access::target AccessTarget>
struct opencl_image_type;

// Creation of dummy ocl types for host_image targets.
// These dummy ocl types are needed by the compiler parser for the compilation
// of host application code with __SYCL_DEVICE_ONLY__ macro set.
template <int Dimensions, access::mode AccessMode>
struct opencl_image_type<Dimensions, AccessMode, access::target::host_image> {
  using type =
      opencl_image_type<Dimensions, AccessMode, access::target::host_image> *;
};
template <typename T> struct sampled_opencl_image_type<T *> {
  using type = void *;
};

#define __SYCL_IMAGETY_DEFINE(Dim, AccessMode, AMSuffix, Target, Ifarray_)     \
  template <>                                                                  \
  struct opencl_image_type<Dim, access::mode::AccessMode,                      \
                           access::target::Target> {                           \
    using type = __ocl_image##Dim##d_##Ifarray_##AMSuffix##_t;                 \
  };
#define __SYCL_SAMPLED_AND_IMAGETY_DEFINE(Dim, AccessMode, AMSuffix, Target,   \
                                          Ifarray_)                            \
  __SYCL_IMAGETY_DEFINE(Dim, AccessMode, AMSuffix, Target, Ifarray_)           \
  template <>                                                                  \
  struct sampled_opencl_image_type<typename opencl_image_type<                 \
      Dim, access::mode::AccessMode, access::target::Target>::type> {          \
    using type = __ocl_sampled_image##Dim##d_##Ifarray_##AMSuffix##_t;         \
  };

#define __SYCL_IMAGETY_READ_3_DIM_IMAGE                                        \
  __SYCL_SAMPLED_AND_IMAGETY_DEFINE(1, read, ro, image, )                      \
  __SYCL_SAMPLED_AND_IMAGETY_DEFINE(2, read, ro, image, )                      \
  __SYCL_SAMPLED_AND_IMAGETY_DEFINE(3, read, ro, image, )

#define __SYCL_IMAGETY_WRITE_3_DIM_IMAGE                                       \
  __SYCL_IMAGETY_DEFINE(1, write, wo, image, )                                 \
  __SYCL_IMAGETY_DEFINE(2, write, wo, image, )                                 \
  __SYCL_IMAGETY_DEFINE(3, write, wo, image, )

#define __SYCL_IMAGETY_DISCARD_WRITE_3_DIM_IMAGE                               \
  __SYCL_IMAGETY_DEFINE(1, discard_write, wo, image, )                         \
  __SYCL_IMAGETY_DEFINE(2, discard_write, wo, image, )                         \
  __SYCL_IMAGETY_DEFINE(3, discard_write, wo, image, )

#define __SYCL_IMAGETY_READ_2_DIM_IARRAY                                       \
  __SYCL_SAMPLED_AND_IMAGETY_DEFINE(1, read, ro, image_array, array_)          \
  __SYCL_SAMPLED_AND_IMAGETY_DEFINE(2, read, ro, image_array, array_)

#define __SYCL_IMAGETY_WRITE_2_DIM_IARRAY                                      \
  __SYCL_IMAGETY_DEFINE(1, write, wo, image_array, array_)                     \
  __SYCL_IMAGETY_DEFINE(2, write, wo, image_array, array_)

#define __SYCL_IMAGETY_DISCARD_WRITE_2_DIM_IARRAY                              \
  __SYCL_IMAGETY_DEFINE(1, discard_write, wo, image_array, array_)             \
  __SYCL_IMAGETY_DEFINE(2, discard_write, wo, image_array, array_)

__SYCL_IMAGETY_READ_3_DIM_IMAGE
__SYCL_IMAGETY_WRITE_3_DIM_IMAGE
__SYCL_IMAGETY_DISCARD_WRITE_3_DIM_IMAGE

__SYCL_IMAGETY_READ_2_DIM_IARRAY
__SYCL_IMAGETY_WRITE_2_DIM_IARRAY
__SYCL_IMAGETY_DISCARD_WRITE_2_DIM_IARRAY

} // namespace detail
} // namespace _V1
} // namespace sycl

#undef __SYCL_SAMPLED_AND_IMAGETY_DEFINE
#undef __SYCL_INVOKE_SPIRV_CALL_ARG1
#undef __SYCL_IMAGETY_DEFINE
#undef __SYCL_IMAGETY_DISCARD_WRITE_3_DIM_IMAGE
#undef __SYCL_IMAGETY_READ_3_DIM_IMAGE
#undef __SYCL_IMAGETY_WRITE_3_DIM_IMAGE
#undef __SYCL_IMAGETY_DISCARD_WRITE_2_DIM_IARRAY
#undef __SYCL_IMAGETY_READ_2_DIM_IARRAY
#undef __SYCL_IMAGETY_WRITE_2_DIM_IARRAY
#endif // #ifdef __SYCL_DEVICE_ONLY__
