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

#include <sycl/__spirv/spirv_ops_image.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {

// Type trait to get the associated sampled image type for a given image type.
template <typename ImageType> struct sampled_opencl_image_type;

// Helper to extract element type from scalar or OpenCL vector type
// (e.g., float or float __attribute__((ext_vector_type(4))))
template <typename T> struct get_image_element_type { using type = T; };
template <typename T, int N>
struct get_image_element_type<T __attribute__((ext_vector_type(N)))> {
  using type = T;
};
template <typename T>
using get_image_element_type_t = typename get_image_element_type<T>::type;

// Helper to get number of elements (1 for scalar, N for vector)
template <typename T> struct get_num_elements {
  static constexpr int value = 1;
};
template <typename T, int N>
struct get_num_elements<T __attribute__((ext_vector_type(N)))> {
  static constexpr int value = N;
};
template <typename T>
inline constexpr int get_num_elements_v = get_num_elements<T>::value;

// The OpenCL SPIR-V environment spec requires that OpImageRead, OpImageWrite,
// OpImageFetch, and OpImageSampleExplicitLod use vec4 operands with 32-bit
// component types, with the sole exception of half (_Float16) which may use
// 16-bit components. Channel sizes and narrow integer types (8-bit and 16-bit)
// must be widened to their vec4 32-bit equivalents.
template <typename T> struct spirv_image_widened_elem_type { using type = T; };
template <> struct spirv_image_widened_elem_type<int8_t>   { using type = int32_t; };
template <> struct spirv_image_widened_elem_type<uint8_t>  { using type = uint32_t; };
template <> struct spirv_image_widened_elem_type<int16_t>  { using type = int32_t; };
template <> struct spirv_image_widened_elem_type<uint16_t> { using type = uint32_t; };
template <typename T>
using spirv_image_widened_elem_type_t = typename spirv_image_widened_elem_type<T>::type;

// Helper function to convert vec4 result to requested OpenCL type.
// Handles scalar, vec2, vec3, and vec4 return types.
template <typename RequestedType, typename Vec4Type>
static inline constexpr RequestedType convertVec4ToRequestedType(Vec4Type vec4Result) {
  using ElemType = get_image_element_type_t<RequestedType>;
  constexpr int NumElements = get_num_elements_v<RequestedType>;

  // Extract components based on RequestedType
  if constexpr (NumElements == 1) {
    return static_cast<ElemType>(vec4Result[0]);
  } else if constexpr (NumElements == 2) {
    using Vec2Type = ElemType __attribute__((ext_vector_type(2)));
    Vec2Type result;
    result[0] = static_cast<ElemType>(vec4Result[0]);
    result[1] = static_cast<ElemType>(vec4Result[1]);
    return result;
  } else if constexpr (NumElements == 3) {
    using Vec3Type = ElemType __attribute__((ext_vector_type(3)));
    Vec3Type result;
    result[0] = static_cast<ElemType>(vec4Result[0]);
    result[1] = static_cast<ElemType>(vec4Result[1]);
    result[2] = static_cast<ElemType>(vec4Result[2]);
    return result;
  } else {
    static_assert(NumElements == 4, "Vector size must be 1, 2, 3, or 4");
    using Vec4NarrowType = ElemType __attribute__((ext_vector_type(4)));
    Vec4NarrowType result;
    result[0] = static_cast<ElemType>(vec4Result[0]);
    result[1] = static_cast<ElemType>(vec4Result[1]);
    result[2] = static_cast<ElemType>(vec4Result[2]);
    result[3] = static_cast<ElemType>(vec4Result[3]);
    return result;
  }
}

// Helper function to convert scalar or OpenCL vector value to vec4 for OpImageWrite.
template <typename SourceType>
static inline constexpr auto convertRequestedTypeToVec4(SourceType val) {
  using RawElemType = get_image_element_type_t<SourceType>;
  using ElemType = spirv_image_widened_elem_type_t<RawElemType>;
  using Vec4Type = ElemType __attribute__((ext_vector_type(4)));
  constexpr int NumElements = get_num_elements_v<SourceType>;

  Vec4Type result{};
  if constexpr (NumElements == 1) {
    result[0] = static_cast<ElemType>(val);
  } else if constexpr (NumElements == 2) {
    result[0] = static_cast<ElemType>(val[0]);
    result[1] = static_cast<ElemType>(val[1]);
  } else if constexpr (NumElements == 3) {
    result[0] = static_cast<ElemType>(val[0]);
    result[1] = static_cast<ElemType>(val[1]);
    result[2] = static_cast<ElemType>(val[2]);
  } else {
    static_assert(NumElements == 4, "Vector size must be 1, 2, 3, or 4");
    result[0] = static_cast<ElemType>(val[0]);
    result[1] = static_cast<ElemType>(val[1]);
    result[2] = static_cast<ElemType>(val[2]);
    result[3] = static_cast<ElemType>(val[3]);
  }
  return result;
}

} // namespace detail
} // namespace _V1
} // namespace sycl

#define __SYCL_INVOKE_SPIRV_CALL_ARG1(call)                                    \
  template <typename R, typename T1> inline R __invoke_##call(T1 ParT1) {      \
    using Ret = sycl::detail::ConvertToOpenCLType_t<R>;                        \
    return sycl::detail::convertFromOpenCLTypeFor<R>(                          \
        __spirv_##call<Ret, T1>(ParT1));                                       \
  }

// The macro defines the function __invoke_ImageXXXX,
// The functions contains the spirv call to __spirv_ImageXXXX.
__SYCL_INVOKE_SPIRV_CALL_ARG1(ImageQuerySize)
__SYCL_INVOKE_SPIRV_CALL_ARG1(ImageQueryFormat)
__SYCL_INVOKE_SPIRV_CALL_ARG1(ImageQueryOrder)

template <typename ImageT, typename CoordT, typename ValT>
static void __invoke__ImageWrite(ImageT Img, CoordT Coords, ValT Val) {

  // Convert from sycl types to builtin types to get correct function mangling.
  auto TmpCoords = sycl::detail::convertToOpenCLType(Coords);
  auto TmpVal = sycl::detail::convertToOpenCLType(Val);

  // SPIR-V spec requires OpImageWrite texel to be vec4.
  auto vec4Val = sycl::detail::convertRequestedTypeToVec4(TmpVal);

  __spirv_ImageWrite<ImageT, decltype(TmpCoords), decltype(vec4Val)>(
      Img, TmpCoords, vec4Val);
}

template <typename RetType, typename ImageT, typename CoordT>
static RetType __invoke__ImageRead(ImageT Img, CoordT Coords) {

  // Convert from sycl types to builtin types to get correct function mangling.
  using TempRetT = sycl::detail::ConvertToOpenCLType_t<RetType>;
  auto TmpCoords = sycl::detail::convertToOpenCLType(Coords);

  // SPIR-V spec requires OpImageRead to return a 32-bit (or 16-bit for half) vec4.
  using RawElemType = sycl::detail::get_image_element_type_t<TempRetT>;
  using ElemType = sycl::detail::spirv_image_widened_elem_type_t<RawElemType>;
  using Vec4Type = ElemType __attribute__((ext_vector_type(4)));

  Vec4Type vec4Result =
      __spirv_ImageRead<Vec4Type, ImageT, decltype(TmpCoords)>(Img, TmpCoords);

  return sycl::detail::convertFromOpenCLTypeFor<RetType>(
      sycl::detail::convertVec4ToRequestedType<TempRetT>(vec4Result));
}

template <typename RetType, typename ImageT, typename CoordT>
static RetType __invoke__ImageFetch(ImageT Img, CoordT Coords) {

  // Convert from sycl types to builtin types to get correct function mangling.
  using TempRetT = sycl::detail::ConvertToOpenCLType_t<RetType>;
  auto TmpCoords = sycl::detail::convertToOpenCLType(Coords);

  // SPIR-V spec requires OpImageFetch to return a 32-bit (or 16-bit for half) vec4.
  using RawElemType = sycl::detail::get_image_element_type_t<TempRetT>;
  using ElemType = sycl::detail::spirv_image_widened_elem_type_t<RawElemType>;
  using Vec4Type = ElemType __attribute__((ext_vector_type(4)));

  Vec4Type vec4Result =
      __spirv_ImageFetch<Vec4Type, ImageT, decltype(TmpCoords)>(Img,
                                                                TmpCoords);

  return sycl::detail::convertFromOpenCLTypeFor<RetType>(
      sycl::detail::convertVec4ToRequestedType<TempRetT>(vec4Result));
}

template <typename RetType, typename ImageT, typename CoordT>
static RetType __invoke__SampledImageFetch(ImageT Img, CoordT Coords) {

  // Convert from sycl types to builtin types to get correct function mangling.
  using TempRetT = sycl::detail::ConvertToOpenCLType_t<RetType>;
  auto TmpCoords = sycl::detail::convertToOpenCLType(Coords);

  // SPIR-V spec requires OpImageFetch to return a 32-bit (or 16-bit for half) vec4.
  using RawElemType = sycl::detail::get_image_element_type_t<TempRetT>;
  using ElemType = sycl::detail::spirv_image_widened_elem_type_t<RawElemType>;
  using Vec4Type = ElemType __attribute__((ext_vector_type(4)));

  Vec4Type vec4Result =
      __spirv_SampledImageFetch<Vec4Type, ImageT, decltype(TmpCoords)>(
          Img, TmpCoords);

  return sycl::detail::convertFromOpenCLTypeFor<RetType>(
      sycl::detail::convertVec4ToRequestedType<TempRetT>(vec4Result));
}

template <typename RetType, typename ImageT, typename CoordT>
static std::enable_if_t<std::is_same_v<RetType, sycl::vec<float, 4>> ||
                            std::is_same_v<RetType, sycl::vec<int, 4>> ||
                            std::is_same_v<RetType, sycl::vec<unsigned int, 4>>,
                        RetType>
__invoke__SampledImageGather(ImageT Img, CoordT Coords, unsigned Component) {

  // Convert from sycl types to builtin types to get correct function mangling.
  using TempRetT = sycl::detail::ConvertToOpenCLType_t<RetType>;
  auto TmpCoords = sycl::detail::convertToOpenCLType(Coords);

  return sycl::detail::convertFromOpenCLTypeFor<RetType>(
      __spirv_SampledImageGather<TempRetT, ImageT, decltype(TmpCoords)>(
          Img, TmpCoords, Component));
}

template <typename RetType, typename ImageT, typename CoordT>
static RetType __invoke__ImageArrayFetch(ImageT Img, CoordT Coords,
                                         int ArrayLayer) {

  // Convert from sycl types to builtin types to get correct function mangling.
  using TempRetT = sycl::detail::ConvertToOpenCLType_t<RetType>;
  auto TmpCoords = sycl::detail::convertToOpenCLType(Coords);

  // SPIR-V spec requires OpImageFetch to return a 32-bit (or 16-bit for half) vec4.
  using RawElemType = sycl::detail::get_image_element_type_t<TempRetT>;
  using ElemType = sycl::detail::spirv_image_widened_elem_type_t<RawElemType>;
  using Vec4Type = ElemType __attribute__((ext_vector_type(4)));

  Vec4Type vec4Result =
      __spirv_ImageArrayFetch<Vec4Type, ImageT, decltype(TmpCoords)>(
          Img, TmpCoords, ArrayLayer);

  return sycl::detail::convertFromOpenCLTypeFor<RetType>(
      sycl::detail::convertVec4ToRequestedType<TempRetT>(vec4Result));
}

template <typename RetType, typename ImageT, typename CoordT>
static RetType __invoke__SampledImageArrayFetch(ImageT Img, CoordT Coords,
                                                int ArrayLayer) {

  // Convert from sycl types to builtin types to get correct function mangling.
  using TempRetT = sycl::detail::ConvertToOpenCLType_t<RetType>;
  auto TmpCoords = sycl::detail::convertToOpenCLType(Coords);

  // SPIR-V spec requires OpImageFetch to return a 32-bit (or 16-bit for half) vec4.
  using RawElemType = sycl::detail::get_image_element_type_t<TempRetT>;
  using ElemType = sycl::detail::spirv_image_widened_elem_type_t<RawElemType>;
  using Vec4Type = ElemType __attribute__((ext_vector_type(4)));

  Vec4Type vec4Result =
      __spirv_SampledImageArrayFetch<Vec4Type, ImageT, decltype(TmpCoords)>(
          Img, TmpCoords, ArrayLayer);

  return sycl::detail::convertFromOpenCLTypeFor<RetType>(
      sycl::detail::convertVec4ToRequestedType<TempRetT>(vec4Result));
}

template <typename RetType, typename ImageT, typename CoordT>
static RetType __invoke__ImageArrayRead(ImageT Img, CoordT Coords,
                                        int ArrayLayer) {

  // Convert from sycl types to builtin types to get correct function mangling.
  using TempRetT = sycl::detail::ConvertToOpenCLType_t<RetType>;
  auto TmpCoords = sycl::detail::convertToOpenCLType(Coords);

  // SPIR-V spec requires OpImageRead to return a 32-bit (or 16-bit for half) vec4.
  using RawElemType = sycl::detail::get_image_element_type_t<TempRetT>;
  using ElemType = sycl::detail::spirv_image_widened_elem_type_t<RawElemType>;
  using Vec4Type = ElemType __attribute__((ext_vector_type(4)));

  Vec4Type vec4Result =
      __spirv_ImageArrayRead<Vec4Type, ImageT, decltype(TmpCoords)>(
          Img, TmpCoords, ArrayLayer);

  return sycl::detail::convertFromOpenCLTypeFor<RetType>(
      sycl::detail::convertVec4ToRequestedType<TempRetT>(vec4Result));
}

template <typename ImageT, typename CoordT, typename ValT>
static void __invoke__ImageArrayWrite(ImageT Img, CoordT Coords, int ArrayLayer,
                                      ValT Val) {

  // Convert from sycl types to builtin types to get correct function mangling.
  auto TmpCoords = sycl::detail::convertToOpenCLType(Coords);
  auto TmpVal = sycl::detail::convertToOpenCLType(Val);

  // SPIR-V spec requires OpImageWrite texel to be vec4.
  auto vec4Val = sycl::detail::convertRequestedTypeToVec4(TmpVal);

  __spirv_ImageArrayWrite<ImageT, decltype(TmpCoords), decltype(vec4Val)>(
      Img, TmpCoords, ArrayLayer, vec4Val);
}

template <typename RetType, typename SmpImageT, typename DirVecT>
static RetType __invoke__ImageReadCubemap(SmpImageT SmpImg, DirVecT DirVec) {

  // Convert from sycl types to builtin types to get correct function mangling.
  using TempRetT = sycl::detail::ConvertToOpenCLType_t<RetType>;
  auto TmpDirVec = sycl::detail::convertToOpenCLType(DirVec);

  // SPIR-V spec requires OpImageSampleExplicitLod to return a 32-bit (or 16-bit for half) vec4.
  using RawElemType = sycl::detail::get_image_element_type_t<TempRetT>;
  using ElemType = sycl::detail::spirv_image_widened_elem_type_t<RawElemType>;
  using Vec4Type = ElemType __attribute__((ext_vector_type(4)));

  Vec4Type vec4Result =
      __spirv_ImageSampleCubemap<SmpImageT, Vec4Type, decltype(TmpDirVec)>(
          SmpImg, TmpDirVec);

  return sycl::detail::convertFromOpenCLTypeFor<RetType>(
      sycl::detail::convertVec4ToRequestedType<TempRetT>(vec4Result));
}

template <typename RetType, typename SmpImageT, typename CoordT>
static RetType __invoke__ImageReadLod(SmpImageT SmpImg, CoordT Coords,
                                      float Level) {
  // Convert from sycl types to builtin types to get correct function mangling.
  using TempRetT = sycl::detail::ConvertToOpenCLType_t<RetType>;
  auto TmpCoords = sycl::detail::convertToOpenCLType(Coords);

  enum ImageOperands { Lod = 0x2 };

  // OpImageSampleExplicitLod
  // Its components must be the same as Sampled Type of the underlying
  // OpTypeImage
  // Sampled Image must be an object whose type is OpTypeSampledImage
  // Image Operands encodes what operands follow. Either Lod
  // or Grad image operands must be present
  // SPIR-V spec requires OpImageSampleExplicitLod to return a 32-bit (or 16-bit for half) vec4.
  using RawElemType = sycl::detail::get_image_element_type_t<TempRetT>;
  using ElemType = sycl::detail::spirv_image_widened_elem_type_t<RawElemType>;
  using Vec4Type = ElemType __attribute__((ext_vector_type(4)));

  Vec4Type vec4Result =
      __spirv_ImageSampleExplicitLod<SmpImageT, Vec4Type, decltype(TmpCoords)>(
          SmpImg, TmpCoords, ImageOperands::Lod, Level);

  return sycl::detail::convertFromOpenCLTypeFor<RetType>(
      sycl::detail::convertVec4ToRequestedType<TempRetT>(vec4Result));
}

template <typename RetType, typename SmpImageT, typename CoordT>
static RetType __invoke__ImageReadGrad(SmpImageT SmpImg, CoordT Coords,
                                       CoordT Dx, CoordT Dy) {

  // Convert from sycl types to builtin types to get correct function mangling.
  using TempRetT = sycl::detail::ConvertToOpenCLType_t<RetType>;
  auto TmpCoords = sycl::detail::convertToOpenCLType(Coords);
  auto TmpGraddX = sycl::detail::convertToOpenCLType(Dx);
  auto TmpGraddY = sycl::detail::convertToOpenCLType(Dy);

  enum ImageOperands { Grad = 0x4 };

  // OpImageSampleExplicitLod
  // Its components must be the same as Sampled Type of the underlying
  // OpTypeImage
  // Sampled Image must be an object whose type is OpTypeSampledImage
  // Image Operands encodes what operands follow. Either Lod
  // or Grad image operands must be present
  // SPIR-V spec requires OpImageSampleExplicitLod to return a 32-bit (or 16-bit for half) vec4.
  using RawElemType = sycl::detail::get_image_element_type_t<TempRetT>;
  using ElemType = sycl::detail::spirv_image_widened_elem_type_t<RawElemType>;
  using Vec4Type = ElemType __attribute__((ext_vector_type(4)));

  Vec4Type vec4Result =
      __spirv_ImageSampleExplicitLod<SmpImageT, Vec4Type, decltype(TmpCoords)>(
          SmpImg, TmpCoords, ImageOperands::Grad, TmpGraddX, TmpGraddY);

  return sycl::detail::convertFromOpenCLTypeFor<RetType>(
      sycl::detail::convertVec4ToRequestedType<TempRetT>(vec4Result));
}

template <typename RetType, typename ImageT, typename CoordT>
static RetType __invoke__ImageReadSampler(ImageT Img, CoordT Coords,
                                          const __ocl_sampler_t &Smpl) {

  // Convert from sycl types to builtin types to get correct function mangling.
  using TempRetT = sycl::detail::ConvertToOpenCLType_t<RetType>;
  using SampledT =
      typename sycl::detail::sampled_opencl_image_type<ImageT>::type;

  auto TmpCoords = sycl::detail::convertToOpenCLType(Coords);
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
  // SPIR-V spec requires OpImageSampleExplicitLod to return a 32-bit (or 16-bit for half) vec4.
  using RawElemType = sycl::detail::get_image_element_type_t<TempRetT>;
  using ElemType = sycl::detail::spirv_image_widened_elem_type_t<RawElemType>;
  using Vec4Type = ElemType __attribute__((ext_vector_type(4)));

  Vec4Type vec4Result =
      __spirv_ImageSampleExplicitLod<SampledT, Vec4Type, decltype(TmpCoords)>(
          __spirv_SampledImage<ImageT, SampledT>(Img, Smpl), TmpCoords,
          ImageOperands::Lod, 0.0f);

  return sycl::detail::convertFromOpenCLTypeFor<RetType>(
      sycl::detail::convertVec4ToRequestedType<TempRetT>(vec4Result));
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
