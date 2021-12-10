//==------------ image_accessor_util.hpp -----------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file includes some utilities that are used by image accessors on host
// device
//

#pragma once

#ifndef __SYCL_DEVICE_ONLY__
#include <CL/sycl/builtins.hpp>
#include <CL/sycl/detail/export.hpp>
#include <CL/sycl/detail/generic_type_traits.hpp>
#include <CL/sycl/image.hpp>
#include <CL/sycl/sampler.hpp>
#include <CL/sycl/types.hpp>

#include <cmath>
#include <iostream>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

template <typename T>
using IsValidCoordType =
    typename is_contained<T, type_list<cl_int, cl_float>>::type;

// The formula for unnormalization coordinates:
// NormalizedCoords = [UnnormalizedCoords[i] * Range[i] for i in range(0, 3)]
template <typename T>
detail::enable_if_t<IsValidCoordType<T>::value, T>
UnnormalizeCoordinates(const T &Coords, const range<3> &Range) {
  return Coords * Range[0];
}

template <typename T>
detail::enable_if_t<IsValidCoordType<T>::value, vec<T, 2>>
UnnormalizeCoordinates(const vec<T, 2> &Coords, const range<3> &Range) {
  return {Coords.x() * Range[0], Coords.y() * Range[1]};
}

template <typename T>
detail::enable_if_t<IsValidCoordType<T>::value, vec<T, 4>>
UnnormalizeCoordinates(const vec<T, 4> &Coords, const range<3> &Range) {
  return {Coords.x() * Range[0], Coords.y() * Range[1], Coords.z() * Range[2],
          0};
}

// Converts the Coordinates from any dimensions into float4.
// valid but unused coordinates are written as 0.5 so the Int_uvwsubhalf
// calculation won't pass 0.
// Non-valid coordinates are written as 0.
template <typename T>
detail::enable_if_t<IsValidCoordType<T>::value, cl_float4>
convertToFloat4(T Coords) {
  return {static_cast<float>(Coords), 0.5f, 0.5f, 0.f};
}

template <typename T>
detail::enable_if_t<IsValidCoordType<T>::value, cl_float4>
convertToFloat4(vec<T, 2> Coords) {
  return {static_cast<float>(Coords.x()), static_cast<float>(Coords.y()), 0.5f,
          0.f};
}

template <typename T>
detail::enable_if_t<IsValidCoordType<T>::value, cl_float4>
convertToFloat4(vec<T, 4> Coords) {
  return {static_cast<float>(Coords.x()), static_cast<float>(Coords.y()),
          static_cast<float>(Coords.z()), 0.f};
}

// This method compute an offset in bytes for a given Coords.
// Retured offset is used to find the address location of a pixel from a base
// ptr.
template <typename T>
detail::enable_if_t<std::is_integral<T>::value, size_t>
getImageOffset(const T &Coords, const id<3>, const uint8_t ElementSize) {
  return Coords * ElementSize;
}

template <typename T>
detail::enable_if_t<std::is_integral<T>::value, size_t>
getImageOffset(const vec<T, 2> &Coords, const id<3> ImgPitch,
               const uint8_t ElementSize) {
  return Coords.x() * ElementSize + Coords.y() * ImgPitch[0];
}

template <typename T>
detail::enable_if_t<std::is_integral<T>::value, size_t>
getImageOffset(const vec<T, 4> &Coords, const id<3> ImgPitch,
               const uint8_t ElementSize) {
  return Coords.x() * ElementSize + Coords.y() * ImgPitch[0] +
         Coords.z() * ImgPitch[1];
}

// Process cl_float4 Coordinates and return the appropriate Pixel Coordinates to
// read from based on Addressing Mode for Nearest filter mode.
__SYCL_EXPORT cl_int4 getPixelCoordNearestFiltMode(cl_float4,
                                                   const addressing_mode,
                                                   const range<3>);

// Process cl_float4 Coordinates and return the appropriate Pixel Coordinates to
// read from based on Addressing Mode for Linear filter mode.
__SYCL_EXPORT cl_int8 getPixelCoordLinearFiltMode(cl_float4,
                                                  const addressing_mode,
                                                  const range<3>, cl_float4 &);

// Check if PixelCoord are out of range for Sampler with clamp adressing mode.
__SYCL_EXPORT bool isOutOfRange(const cl_int4 PixelCoord,
                                const addressing_mode SmplAddrMode,
                                const range<3> ImgRange);

// Get Border Color for the image_channel_order, the border color values are
// only used when the sampler has clamp addressing mode.
__SYCL_EXPORT cl_float4
getBorderColor(const image_channel_order ImgChannelOrder);

// Reads data from a pixel at Ptr location, based on the number of Channels in
// Order and returns the data.
// The datatype used to read from the Ptr is based on the T of the
// image. This datatype is computed by the calling API.
template <typename T>
vec<T, 4> readPixel(T *Ptr, const image_channel_order ChannelOrder,
                    const image_channel_type ChannelType) {
  vec<T, 4> Pixel(0);

  switch (ChannelOrder) {
  case image_channel_order::a:
    Pixel.w() = Ptr[0];
    break;
  case image_channel_order::r:
  case image_channel_order::rx:
    Pixel.x() = Ptr[0];
    Pixel.w() = 1;
    break;
  case image_channel_order::intensity:
    Pixel.x() = Ptr[0];
    Pixel.y() = Ptr[0];
    Pixel.z() = Ptr[0];
    Pixel.w() = Ptr[0];
    break;
  case image_channel_order::luminance:
    Pixel.x() = Ptr[0];
    Pixel.y() = Ptr[0];
    Pixel.z() = Ptr[0];
    Pixel.w() = 1.0;
    break;
  case image_channel_order::rg:
  case image_channel_order::rgx:
    Pixel.x() = Ptr[0];
    Pixel.y() = Ptr[1];
    Pixel.w() = 1.0;
    break;
  case image_channel_order::ra:
    Pixel.x() = Ptr[0];
    Pixel.w() = Ptr[1];
    break;
  case image_channel_order::rgb:
  case image_channel_order::rgbx:
    if (ChannelType == image_channel_type::unorm_short_565 ||
        ChannelType == image_channel_type::unorm_short_555 ||
        ChannelType == image_channel_type::unorm_int_101010) {
      Pixel.x() = Ptr[0];
    } else {
      Pixel.x() = Ptr[0];
      Pixel.y() = Ptr[1];
      Pixel.z() = Ptr[2];
      Pixel.w() = 1.0;
    }
    break;
  case image_channel_order::rgba:
  case image_channel_order::ext_oneapi_srgba:
    Pixel.x() = Ptr[0]; // r
    Pixel.y() = Ptr[1]; // g
    Pixel.z() = Ptr[2]; // b
    Pixel.w() = Ptr[3]; // a
    break;
  case image_channel_order::argb:
    Pixel.w() = Ptr[0]; // a
    Pixel.x() = Ptr[1]; // r
    Pixel.y() = Ptr[2]; // g
    Pixel.z() = Ptr[3]; // b
    break;
  case image_channel_order::bgra:
    Pixel.z() = Ptr[0]; // b
    Pixel.y() = Ptr[1]; // g
    Pixel.x() = Ptr[2]; // r
    Pixel.w() = Ptr[3]; // a
    break;
  case image_channel_order::abgr:
    Pixel.w() = Ptr[0]; // a
    Pixel.z() = Ptr[1]; // b
    Pixel.y() = Ptr[2]; // g
    Pixel.x() = Ptr[3]; // r
    break;
  }

  return Pixel;
}

// Write data to a pixel at Ptr location, based on the number of Channels in
// ImageChannelOrder. The data passed to this API in 'Pixel' is already
// converted to Datatype of the Channel based on ImageChannelType by the calling
// API.
template <typename T>
void writePixel(const vec<T, 4> Pixel, T *Ptr,
                const image_channel_order ChannelOrder,
                const image_channel_type ChannelType) {

  // Data is written based on section 6.12.14.6 of openCL spec.
  switch (ChannelOrder) {
  case image_channel_order::a:
    Ptr[0] = Pixel.w();
    break;
  case image_channel_order::r:
  case image_channel_order::rx:
  case image_channel_order::intensity:
  case image_channel_order::luminance:
    Ptr[0] = Pixel.x();
    break;
  case image_channel_order::rg:
  case image_channel_order::rgx:
    Ptr[0] = Pixel.x();
    Ptr[1] = Pixel.y();
    break;
  case image_channel_order::ra:
    Ptr[0] = Pixel.x();
    Ptr[1] = Pixel.w();
    break;
  case image_channel_order::rgb:
  case image_channel_order::rgbx:
    if (ChannelType == image_channel_type::unorm_short_565 ||
        ChannelType == image_channel_type::unorm_short_555 ||
        ChannelType == image_channel_type::unorm_int_101010) {
      Ptr[0] = Pixel.x();
    } else {
      Ptr[0] = Pixel.x();
      Ptr[1] = Pixel.y();
      Ptr[2] = Pixel.z();
    }
    break;
  case image_channel_order::rgba:
  case image_channel_order::ext_oneapi_srgba:
    Ptr[0] = Pixel.x(); // r
    Ptr[1] = Pixel.y(); // g
    Ptr[2] = Pixel.z(); // b
    Ptr[3] = Pixel.w(); // a
    break;
  case image_channel_order::argb:
    Ptr[0] = Pixel.w(); // a
    Ptr[1] = Pixel.x(); // r
    Ptr[2] = Pixel.y(); // g
    Ptr[3] = Pixel.z(); // b
    break;
  case image_channel_order::bgra:
    Ptr[0] = Pixel.z(); // b
    Ptr[1] = Pixel.y(); // g
    Ptr[2] = Pixel.x(); // r
    Ptr[3] = Pixel.w(); // a
    break;
  case image_channel_order::abgr:
    Ptr[0] = Pixel.w(); // a
    Ptr[1] = Pixel.z(); // b
    Ptr[2] = Pixel.y(); // g
    Ptr[3] = Pixel.x(); // r
    break;
  }
}

// Converts read pixel data into return datatype based on the channel type of
// the image.
// Conversion rules used as given in the OpenCL
// Spec section 8.3. The conversion rules may be handled differently for each
// return datatype - float, int32, uint32, half. ImageChannelType is passed to
// the function to use appropriate conversion rules.

template <typename ChannelType>
void convertReadData(const vec<ChannelType, 4> PixelData,
                     const image_channel_type ImageChannelType,
                     vec<cl_uint, 4> &RetData) {

  switch (ImageChannelType) {
  case image_channel_type::unsigned_int8:
  case image_channel_type::unsigned_int16:
  case image_channel_type::unsigned_int32:
    RetData = PixelData.template convert<cl_uint>();
    break;
  default:
    // OpenCL Spec section 6.12.14.2 does not allow reading uint4 data from an
    // image with channel datatype other than unsigned_int8,unsigned_int16 and
    // unsigned_int32.
    throw cl::sycl::invalid_parameter_error(
        "Datatype of read data - cl_uint4 is incompatible with the "
        "image_channel_type of the image.",
        PI_INVALID_VALUE);
  }
}

template <typename ChannelType>
void convertReadData(const vec<ChannelType, 4> PixelData,
                     const image_channel_type ImageChannelType,
                     vec<cl_int, 4> &RetData) {

  switch (ImageChannelType) {
  case image_channel_type::signed_int8:
  case image_channel_type::signed_int16:
  case image_channel_type::signed_int32:
    RetData = PixelData.template convert<cl_int>();
    break;
  default:
    // OpenCL Spec section 6.12.14.2 does not allow reading int4 data from an
    // image with channel datatype other than signed_int8,signed_int16 and
    // signed_int32.
    throw cl::sycl::invalid_parameter_error(
        "Datatype of read data - cl_int4 is incompatible with the "
        "image_channel_type of the image.",
        PI_INVALID_VALUE);
  }
}

template <typename ChannelType>
void convertReadData(const vec<ChannelType, 4> PixelData,
                     const image_channel_type ImageChannelType,
                     vec<cl_float, 4> &RetData) {

  switch (ImageChannelType) {
  case image_channel_type::snorm_int8:
    //  max(-1.0f, (float)c / 127.0f)
    RetData = (PixelData.template convert<cl_float>()) / 127.0f;
    RetData = cl::sycl::fmax(RetData, -1);
    break;
  case image_channel_type::snorm_int16:
    // max(-1.0f, (float)c / 32767.0f)
    RetData = (PixelData.template convert<cl_float>()) / 32767.0f;
    RetData = cl::sycl::fmax(RetData, -1);
    break;
  case image_channel_type::unorm_int8:
    // (float)c / 255.0f
    RetData = (PixelData.template convert<cl_float>()) / 255.0f;
    break;
  case image_channel_type::unorm_int16:
    // (float)c / 65535.0f
    RetData = (PixelData.template convert<cl_float>()) / 65535.0f;
    break;
  case image_channel_type::unorm_short_565: {
    // TODO: Missing information in OpenCL spec. check if the below code is
    // correct after the spec is updated.
    // Assuming: (float)c / 31.0f; c represents the 5-bit integer.
    //           (float)c / 63.0f; c represents the 6-bit integer.
    // PixelData.x will be of type cl_ushort.
    vec<cl_ushort, 4> Temp(PixelData.x());
    vec<cl_ushort, 4> MaskBits(0xF800 /*r:bits 11-15*/, 0x07E0 /*g:bits 5-10*/,
                               0x001F /*b:bits 0-4*/, 0x0000);
    vec<cl_ushort, 4> ShiftBits(11, 5, 0, 0);
    vec<cl_float, 4> DivisorToNormalise(31.0f, 63.0f, 31.0f, 1);
    Temp = (Temp & MaskBits) >> ShiftBits;
    RetData = (Temp.template convert<cl_float>()) / DivisorToNormalise;
    break;
  }
  case image_channel_type::unorm_short_555: {
    // TODO: Missing information in OpenCL spec. check if the below code is
    // correct after the spec is updated.
    // Assuming: (float)c / 31.0f; c represents the 5-bit integer.

    // Extracting each 5-bit channel data.
    // PixelData.x will be of type cl_ushort.
    vec<cl_ushort, 4> Temp(PixelData.x());
    vec<cl_ushort, 4> MaskBits(0x7C00 /*r:bits 10-14*/, 0x03E0 /*g:bits 5-9*/,
                               0x001F /*b:bits 0-4*/, 0x0000);
    vec<cl_ushort, 4> ShiftBits(10, 5, 0, 0);
    Temp = (Temp & MaskBits) >> ShiftBits;
    RetData = (Temp.template convert<cl_float>()) / 31.0f;
    break;
  }
  case image_channel_type::unorm_int_101010: {
    // Extracting each 10-bit channel data.
    // PixelData.x will be of type cl_uint.
    vec<cl_uint, 4> Temp(PixelData.x());
    vec<cl_uint, 4> MaskBits(0x3FF00000 /*r:bits 20-29*/,
                             0x000FFC00 /*g:bits 10-19*/,
                             0x000003FF /*b:bits 0-9*/, 0x00000000);
    vec<cl_uint, 4> ShiftBits(20, 10, 0, 0);
    Temp = (Temp & MaskBits) >> ShiftBits;
    RetData = (Temp.template convert<cl_float>()) / 1023.0f;
    break;
  }
  case image_channel_type::signed_int8:
  case image_channel_type::signed_int16:
  case image_channel_type::signed_int32:
  case image_channel_type::unsigned_int8:
  case image_channel_type::unsigned_int16:
  case image_channel_type::unsigned_int32:
    // OpenCL Spec section 6.12.14.2 does not allow reading float4 data from an
    // image with channel datatype -  signed/unsigned_int8,signed/unsigned_int16
    // and signed/unsigned_int32.
    throw cl::sycl::invalid_parameter_error(
        "Datatype of read data - cl_float4 is incompatible with the "
        "image_channel_type of the image.",
        PI_INVALID_VALUE);
  case image_channel_type::fp16:
    // Host has conversion from float to half with accuracy as required in
    // section 8.3.2 OpenCL spec.
    RetData = PixelData.template convert<cl_float>();
    break;
  case image_channel_type::fp32:
    RetData = PixelData.template convert<cl_float>();
    break;
  }
}

template <typename ChannelType>
void convertReadData(const vec<ChannelType, 4> PixelData,
                     const image_channel_type ImageChannelType,
                     vec<cl_half, 4> &RetData) {
  vec<cl_float, 4> RetDataFloat;
  switch (ImageChannelType) {
  case image_channel_type::snorm_int8:
    //  max(-1.0f, (half)c / 127.0f)
    RetDataFloat = (PixelData.template convert<cl_float>()) / 127.0f;
    RetDataFloat = cl::sycl::fmax(RetDataFloat, -1);
    break;
  case image_channel_type::snorm_int16:
    // max(-1.0f, (half)c / 32767.0f)
    RetDataFloat = (PixelData.template convert<cl_float>()) / 32767.0f;
    RetDataFloat = cl::sycl::fmax(RetDataFloat, -1);
    break;
  case image_channel_type::unorm_int8:
    // (half)c / 255.0f
    RetDataFloat = (PixelData.template convert<cl_float>()) / 255.0f;
    break;
  case image_channel_type::unorm_int16:
    // (half)c / 65535.0f
    RetDataFloat = (PixelData.template convert<cl_float>()) / 65535.0f;
    break;
  case image_channel_type::unorm_short_565:
  case image_channel_type::unorm_short_555:
  case image_channel_type::unorm_int_101010:
    // TODO: Missing information in OpenCL spec.
    throw cl::sycl::feature_not_supported(
        "Currently unsupported datatype conversion from image_channel_type "
        "to cl_half4.",
        PI_INVALID_OPERATION);
  case image_channel_type::signed_int8:
  case image_channel_type::signed_int16:
  case image_channel_type::signed_int32:
  case image_channel_type::unsigned_int8:
  case image_channel_type::unsigned_int16:
  case image_channel_type::unsigned_int32:
    // OpenCL Spec section 6.12.14.2 does not allow reading float4 data to an
    // image with channel datatype - signed/unsigned_int8,signed/unsigned_int16
    // and signed/unsigned_int32.
    throw cl::sycl::invalid_parameter_error(
        "Datatype to read- cl_half4 is incompatible with the "
        "image_channel_type of the image.",
        PI_INVALID_VALUE);
  case image_channel_type::fp16:
    RetData = PixelData.template convert<cl_half>();
    return;
  case image_channel_type::fp32:
    throw cl::sycl::invalid_parameter_error(
        "Datatype to read - cl_half4 is incompatible with the "
        "image_channel_type of the image.",
        PI_INVALID_VALUE);
  }
  RetData = RetDataFloat.template convert<cl_half>();
}

// Converts data to write into appropriate datatype based on the channel of the
// image.
// The conversion rules used are as given in OpenCL Spec Section 8.3. The
// conversion rules are different for each return datatype - float,
// int32, uint32, half. ImageChannelType is passed to the function to use
// appropriate conversion rules.
template <typename ChannelType>
vec<ChannelType, 4>
convertWriteData(const vec<cl_uint, 4> WriteData,
                 const image_channel_type ImageChannelType) {
  switch (ImageChannelType) {
  case image_channel_type::unsigned_int8: {
    // convert_uchar_sat(Data)
    cl_uint MinVal = min_v<cl_uchar>();
    cl_uint MaxVal = max_v<cl_uchar>();
    vec<cl_uint, 4> PixelData = cl::sycl::clamp(WriteData, MinVal, MaxVal);
    return PixelData.convert<ChannelType>();
  }
  case image_channel_type::unsigned_int16: {
    // convert_ushort_sat(Data)
    cl_uint MinVal = min_v<cl_ushort>();
    cl_uint MaxVal = max_v<cl_ushort>();
    vec<cl_uint, 4> PixelData = cl::sycl::clamp(WriteData, MinVal, MaxVal);
    return PixelData.convert<ChannelType>();
  }
  case image_channel_type::unsigned_int32:
    // no conversion is performed.
    return WriteData.convert<ChannelType>();
  default:
    // OpenCL Spec section 6.12.14.4 does not allow writing uint4 data to an
    // image with channel datatype other than unsigned_int8,unsigned_int16 and
    // unsigned_int32.
    throw cl::sycl::invalid_parameter_error(
        "Datatype of data to write - cl_uint4 is incompatible with the "
        "image_channel_type of the image.",
        PI_INVALID_VALUE);
  }
}

template <typename ChannelType>
vec<ChannelType, 4>
convertWriteData(const vec<cl_int, 4> WriteData,
                 const image_channel_type ImageChannelType) {

  switch (ImageChannelType) {
  case image_channel_type::signed_int8: {
    // convert_char_sat(Data)
    cl_int MinVal = min_v<cl_char>();
    cl_int MaxVal = max_v<cl_char>();
    vec<cl_int, 4> PixelData = cl::sycl::clamp(WriteData, MinVal, MaxVal);
    return PixelData.convert<ChannelType>();
  }
  case image_channel_type::signed_int16: {
    // convert_short_sat(Data)
    cl_int MinVal = min_v<cl_short>();
    cl_int MaxVal = max_v<cl_short>();
    vec<cl_int, 4> PixelData = cl::sycl::clamp(WriteData, MinVal, MaxVal);
    return PixelData.convert<ChannelType>();
  }
  case image_channel_type::signed_int32:
    return WriteData.convert<ChannelType>();
  default:
    // OpenCL Spec section 6.12.14.4 does not allow writing int4 data to an
    // image with channel datatype other than signed_int8,signed_int16 and
    // signed_int32.
    throw cl::sycl::invalid_parameter_error(
        "Datatype of data to write - cl_int4 is incompatible with the "
        "image_channel_type of the image.",
        PI_INVALID_VALUE);
  }
}

template <typename ChannelType>
vec<ChannelType, 4> processFloatDataToPixel(vec<cl_float, 4> WriteData,
                                            float MulFactor) {
  vec<cl_float, 4> Temp = WriteData * MulFactor;
  vec<cl_int, 4> TempInInt = Temp.convert<int, rounding_mode::rte>();
  vec<cl_int, 4> TempInIntSaturated =
      cl::sycl::clamp(TempInInt, min_v<ChannelType>(), max_v<ChannelType>());
  return TempInIntSaturated.convert<ChannelType>();
}

template <typename ChannelType>
vec<ChannelType, 4>
convertWriteData(const vec<cl_float, 4> WriteData,
                 const image_channel_type ImageChannelType) {

  vec<ChannelType, 4> PixelData;

  switch (ImageChannelType) {
  case image_channel_type::snorm_int8:
    // convert_char_sat_rte(f * 127.0f)
    return processFloatDataToPixel<ChannelType>(WriteData, 127.0f);
  case image_channel_type::snorm_int16:
    // convert_short_sat_rte(f * 32767.0f)
    return processFloatDataToPixel<ChannelType>(WriteData, 32767.0f);
  case image_channel_type::unorm_int8:
    // convert_uchar_sat_rte(f * 255.0f)
    return processFloatDataToPixel<ChannelType>(WriteData, 255.0f);
  case image_channel_type::unorm_int16:
    // convert_ushort_sat_rte(f * 65535.0f)
    return processFloatDataToPixel<ChannelType>(WriteData, 65535.0f);
  case image_channel_type::unorm_short_565:
    // TODO: Missing information in OpenCL spec.
    throw cl::sycl::feature_not_supported(
        "Currently unsupported datatype conversion from image_channel_type "
        "to cl_float4.",
        PI_INVALID_OPERATION);
  case image_channel_type::unorm_short_555:
    // TODO: Missing information in OpenCL spec.
    // Check if the below code is correct after the spec is updated.
    // Assuming: min(convert_ushort_sat_rte(f * 32.0f), 0x1f)
    // bits 9:5 and B in bits 4:0.
    {
      vec<cl_ushort, 4> PixelData =
          processFloatDataToPixel<cl_ushort>(WriteData, 32.0f);
      PixelData = cl::sycl::min(PixelData, static_cast<ChannelType>(0x1f));
      // Compressing the data into the first element of PixelData.
      // This is needed so that the data can be directly stored into the pixel
      // location from the first element.
      // For CL_UNORM_SHORT_555, bit 15 is undefined, R is in bits 14:10, G
      // in bits 9:5 and B in bits 4:0
      PixelData.x() =
          (PixelData.x() << 10) | (PixelData.y() << 5) | PixelData.z();
      return PixelData.convert<ChannelType>();
    }
  case image_channel_type::unorm_int_101010:
    // min(convert_ushort_sat_rte(f * 1023.0f), 0x3ff)
    // For CL_UNORM_INT_101010, bits 31:30 are undefined, R is in bits 29:20, G
    // in bits 19:10 and B in bits 9:0
    {
      vec<cl_uint, 4> PixelData =
          processFloatDataToPixel<cl_uint>(WriteData, 1023.0f);
      PixelData = cl::sycl::min(PixelData, static_cast<ChannelType>(0x3ff));
      PixelData.x() =
          (PixelData.x() << 20) | (PixelData.y() << 10) | PixelData.z();
      return PixelData.convert<ChannelType>();
    }
  case image_channel_type::signed_int8:
  case image_channel_type::signed_int16:
  case image_channel_type::signed_int32:
  case image_channel_type::unsigned_int8:
  case image_channel_type::unsigned_int16:
  case image_channel_type::unsigned_int32:
    // OpenCL Spec section 6.12.14.4 does not allow writing float4 data to an
    // image with channel datatype -  signed/unsigned_int8,signed/unsigned_int16
    // and signed/unsigned_int32.
    throw cl::sycl::invalid_parameter_error(
        "Datatype of data to write - cl_float4 is incompatible with the "
        "image_channel_type of the image.",
        PI_INVALID_VALUE);
  case image_channel_type::fp16:
    // Host has conversion from float to half with accuracy as required in
    // section 8.3.2 OpenCL spec.
    return WriteData.convert<ChannelType>();
  case image_channel_type::fp32:
    return WriteData.convert<ChannelType>();
  }
}

template <typename ChannelType>
vec<ChannelType, 4>
convertWriteData(const vec<cl_half, 4> WriteData,
                 const image_channel_type ImageChannelType) {
  vec<cl_float, 4> WriteDataFloat = WriteData.convert<cl_float>();
  switch (ImageChannelType) {
  case image_channel_type::snorm_int8:
    // convert_char_sat_rte(h * 127.0f)
    return processFloatDataToPixel<ChannelType>(WriteDataFloat, 127.0f);
  case image_channel_type::snorm_int16:
    // convert_short_sat_rte(h * 32767.0f)
    return processFloatDataToPixel<ChannelType>(WriteDataFloat, 32767.0f);
  case image_channel_type::unorm_int8:
    // convert_uchar_sat_rte(h * 255.0f)
    return processFloatDataToPixel<ChannelType>(WriteDataFloat, 255.0f);
  case image_channel_type::unorm_int16:
    // convert_ushort_sat_rte(h * 65535.0f)
    return processFloatDataToPixel<ChannelType>(WriteDataFloat, 65535.0f);
  case image_channel_type::unorm_short_565:
  case image_channel_type::unorm_short_555:
  case image_channel_type::unorm_int_101010:
    // TODO: Missing information in OpenCL spec.
    throw cl::sycl::feature_not_supported(
        "Currently unsupported datatype conversion from image_channel_type "
        "to cl_half4.",
        PI_INVALID_OPERATION);
  case image_channel_type::signed_int8:
  case image_channel_type::signed_int16:
  case image_channel_type::signed_int32:
  case image_channel_type::unsigned_int8:
  case image_channel_type::unsigned_int16:
  case image_channel_type::unsigned_int32:
    // OpenCL Spec section 6.12.14.4 does not allow writing float4 data to an
    // image with channel datatype - signed/unsigned_int8,signed/unsigned_int16
    // and signed/unsigned_int32.
    throw cl::sycl::invalid_parameter_error(
        "Datatype of data to write - cl_float4 is incompatible with the "
        "image_channel_type of the image.",
        PI_INVALID_VALUE);
  case image_channel_type::fp16:
    return WriteData.convert<ChannelType>();
  case image_channel_type::fp32:
    throw cl::sycl::invalid_parameter_error(
        "Datatype of data to write - cl_float4 is incompatible with the "
        "image_channel_type of the image.",
        PI_INVALID_VALUE);
  }
}

// imageWriteHostImpl method is called by the write API in image accessors for
// host device. Steps:
// 1. Calculates the offset from the base ptr of the image where the pixel
// denoted by Coord is located.(getImageOffset method.)
// 2. Converts the ptr to the appropriate datatype based on
// ImageChannelType.(reinterpret_cast)
// 3. The data is converted to the image pixel data based on conversion rules in
// the spec.(convertWriteData)
// 4. The converted data is then written to the pixel at Ptr, based on Number of
// Channels in the Image.(writePixel)
// Note: We assume that Coords are in the appropriate image range. OpenCL
// Spec says that the behaviour is undefined when the Coords are passed outside
// the image range. In the current implementation, the data gets written to the
// calculated Ptr.
template <typename CoordT, typename WriteDataT>
void imageWriteHostImpl(const CoordT &Coords, const WriteDataT &Color,
                        id<3> ImgPitch, uint8_t ElementSize,
                        image_channel_type ImgChannelType,
                        image_channel_order ImgChannelOrder, void *BasePtr) {
  // Calculate position to write
  auto Ptr = static_cast<unsigned char *>(BasePtr) +
             getImageOffset(Coords, ImgPitch, ElementSize);

  switch (ImgChannelType) {
  case image_channel_type::snorm_int8:
    writePixel(convertWriteData<cl_char>(Color, ImgChannelType),
               reinterpret_cast<cl_char *>(Ptr), ImgChannelOrder,
               ImgChannelType);
    break;
  case image_channel_type::snorm_int16:
    writePixel(convertWriteData<cl_short>(Color, ImgChannelType),
               reinterpret_cast<cl_short *>(Ptr), ImgChannelOrder,
               ImgChannelType);
    break;
  case image_channel_type::unorm_int8:
    writePixel(convertWriteData<cl_uchar>(Color, ImgChannelType),
               reinterpret_cast<cl_uchar *>(Ptr), ImgChannelOrder,
               ImgChannelType);
    break;
  case image_channel_type::unorm_int16:
    writePixel(convertWriteData<cl_ushort>(Color, ImgChannelType),
               reinterpret_cast<cl_ushort *>(Ptr), ImgChannelOrder,
               ImgChannelType);
    break;
  case image_channel_type::unorm_short_565:
    writePixel(convertWriteData<short>(Color, ImgChannelType),
               reinterpret_cast<short *>(Ptr), ImgChannelOrder, ImgChannelType);
    break;
  case image_channel_type::unorm_short_555:
    writePixel(convertWriteData<short>(Color, ImgChannelType),
               reinterpret_cast<short *>(Ptr), ImgChannelOrder, ImgChannelType);
    break;
  case image_channel_type::unorm_int_101010:
    writePixel(convertWriteData<cl_uint>(Color, ImgChannelType),
               reinterpret_cast<cl_uint *>(Ptr), ImgChannelOrder,
               ImgChannelType);
    break;
  case image_channel_type::signed_int8:
    writePixel(convertWriteData<cl_char>(Color, ImgChannelType),
               reinterpret_cast<cl_char *>(Ptr), ImgChannelOrder,
               ImgChannelType);
    break;
  case image_channel_type::signed_int16:
    writePixel(convertWriteData<cl_short>(Color, ImgChannelType),
               reinterpret_cast<cl_short *>(Ptr), ImgChannelOrder,
               ImgChannelType);
    break;
  case image_channel_type::signed_int32:
    writePixel(convertWriteData<cl_int>(Color, ImgChannelType),
               reinterpret_cast<cl_int *>(Ptr), ImgChannelOrder,
               ImgChannelType);
    break;
  case image_channel_type::unsigned_int8:
    writePixel(convertWriteData<cl_uchar>(Color, ImgChannelType),
               reinterpret_cast<cl_uchar *>(Ptr), ImgChannelOrder,
               ImgChannelType);
    break;
  case image_channel_type::unsigned_int16:
    writePixel(convertWriteData<cl_ushort>(Color, ImgChannelType),
               reinterpret_cast<cl_ushort *>(Ptr), ImgChannelOrder,
               ImgChannelType);
    break;
  case image_channel_type::unsigned_int32:
    writePixel(convertWriteData<cl_uint>(Color, ImgChannelType),
               reinterpret_cast<cl_uint *>(Ptr), ImgChannelOrder,
               ImgChannelType);
    break;
  case image_channel_type::fp16:
    writePixel(
        // convertWriteDataToHalf<typename
        // TryToGetElementType<WriteDataT>::type>(
        convertWriteData<cl_half>(Color, ImgChannelType),
        reinterpret_cast<cl_half *>(Ptr), ImgChannelOrder, ImgChannelType);
    break;
  case image_channel_type::fp32:
    writePixel(convertWriteData<cl_float>(Color, ImgChannelType),
               reinterpret_cast<cl_float *>(Ptr), ImgChannelOrder,
               ImgChannelType);
    break;
  }
}

// Method called to read a Coord by getColor function when the Coord is
// in-range. This method takes Unnormalized Coords - 'PixelCoord' as cl_int4.
// Invalid Coord are denoted by 0. Steps:
// 1. Compute Offset for given Unnormalised Coordinates using ImagePitch and
// ElementSize.(getImageOffset)
// 2. Add this Offset to BasePtr to compute the location of the Image.
// 3. Convert this Ptr to the appropriate datatype pointer based on
// ImageChannelType. (reinterpret_cast)
// 4. Read the appropriate number of channels(computed using
// ImageChannelOrder) of the appropriate Channel datatype into Color
// variable.(readPixel)
// 5. Convert the Read Data into Return DataT based on conversion rules in
// the Spec.(convertReadData)
// Possible DataT are cl_int4, cl_uint4, cl_float4, cl_half4;
template <typename DataT>
DataT ReadPixelData(const cl_int4 PixelCoord, const id<3> ImgPitch,
                    const image_channel_type ImageChannelType,
                    const image_channel_order ImageChannelOrder,
                    void *BasePtr, const uint8_t ElementSize) {
  DataT Color(0);
  auto Ptr = static_cast<unsigned char *>(BasePtr) +
             getImageOffset(PixelCoord, ImgPitch,
                            ElementSize); // Utility to compute offset in
                                          // image_accessor_util.hpp

  switch (ImageChannelType) {
    // TODO: Pass either ImageChannelType or the exact channel type to the
    // readPixel Function.
  case image_channel_type::snorm_int8:
    convertReadData<cl_char>(readPixel(reinterpret_cast<cl_char *>(Ptr),
                                       ImageChannelOrder, ImageChannelType),
                             image_channel_type::snorm_int8, Color);
    break;
  case image_channel_type::snorm_int16:
    convertReadData<cl_short>(readPixel(reinterpret_cast<cl_short *>(Ptr),
                                        ImageChannelOrder, ImageChannelType),
                              image_channel_type::snorm_int16, Color);
    break;
  case image_channel_type::unorm_int8:
    convertReadData<cl_uchar>(readPixel(reinterpret_cast<cl_uchar *>(Ptr),
                                        ImageChannelOrder, ImageChannelType),
                              image_channel_type::unorm_int8, Color);
    break;
  case image_channel_type::unorm_int16:
    convertReadData<cl_ushort>(readPixel(reinterpret_cast<cl_ushort *>(Ptr),
                                         ImageChannelOrder, ImageChannelType),
                               image_channel_type::unorm_int16, Color);
    break;
  case image_channel_type::unorm_short_565:
    convertReadData<cl_ushort>(readPixel(reinterpret_cast<cl_ushort *>(Ptr),
                                         ImageChannelOrder, ImageChannelType),
                               image_channel_type::unorm_short_565, Color);
    break;
  case image_channel_type::unorm_short_555:
    convertReadData<cl_ushort>(readPixel(reinterpret_cast<cl_ushort *>(Ptr),
                                         ImageChannelOrder, ImageChannelType),
                               image_channel_type::unorm_short_555, Color);
    break;
  case image_channel_type::unorm_int_101010:
    convertReadData<cl_uint>(readPixel(reinterpret_cast<cl_uint *>(Ptr),
                                       ImageChannelOrder, ImageChannelType),
                             image_channel_type::unorm_int_101010, Color);
    break;
  case image_channel_type::signed_int8:
    convertReadData<cl_char>(readPixel(reinterpret_cast<cl_char *>(Ptr),
                                       ImageChannelOrder, ImageChannelType),
                             image_channel_type::signed_int8, Color);
    break;
  case image_channel_type::signed_int16:
    convertReadData<cl_short>(readPixel(reinterpret_cast<cl_short *>(Ptr),
                                        ImageChannelOrder, ImageChannelType),
                              image_channel_type::signed_int16, Color);
    break;
  case image_channel_type::signed_int32:
    convertReadData<cl_int>(readPixel(reinterpret_cast<cl_int *>(Ptr),
                                      ImageChannelOrder, ImageChannelType),
                            image_channel_type::signed_int32, Color);
    break;
  case image_channel_type::unsigned_int8:
    convertReadData<cl_uchar>(readPixel(reinterpret_cast<cl_uchar *>(Ptr),
                                        ImageChannelOrder, ImageChannelType),
                              image_channel_type::unsigned_int8, Color);
    break;
  case image_channel_type::unsigned_int16:
    convertReadData<cl_ushort>(readPixel(reinterpret_cast<cl_ushort *>(Ptr),
                                         ImageChannelOrder, ImageChannelType),
                               image_channel_type::unsigned_int16, Color);
    break;
  case image_channel_type::unsigned_int32:
    convertReadData<cl_uint>(readPixel(reinterpret_cast<cl_uint *>(Ptr),
                                       ImageChannelOrder, ImageChannelType),
                             image_channel_type::unsigned_int32, Color);
    break;
  case image_channel_type::fp16:
    convertReadData<cl_half>(readPixel(reinterpret_cast<cl_half *>(Ptr),
                                       ImageChannelOrder, ImageChannelType),
                             image_channel_type::fp16, Color);
    break;
  case image_channel_type::fp32:
    convertReadData<cl_float>(readPixel(reinterpret_cast<cl_float *>(Ptr),
                                        ImageChannelOrder, ImageChannelType),
                              image_channel_type::fp32, Color);
    break;
  }

  return Color;
}

// Checks if the PixelCoord is out-of-range, and returns appropriate border or
// color value at the PixelCoord.
template <typename DataT>
DataT getColor(const cl_int4 PixelCoord, const addressing_mode SmplAddrMode,
               const range<3> ImgRange, const id<3> ImgPitch,
               const image_channel_type ImgChannelType,
               const image_channel_order ImgChannelOrder, void *BasePtr,
               const uint8_t ElementSize) {
  DataT RetData;
  if (isOutOfRange(PixelCoord, SmplAddrMode, ImgRange)) {
    cl_float4 BorderColor = getBorderColor(ImgChannelOrder);
    RetData = BorderColor.convert<typename TryToGetElementType<DataT>::type>();
  } else {
    RetData = ReadPixelData<DataT>(PixelCoord, ImgPitch, ImgChannelType,
                                   ImgChannelOrder, BasePtr, ElementSize);
  }
  return RetData;
}

// Computes and returns color value with Linear Filter Mode.
// Steps:
// 1. Computes the 8 coordinates using all combinations of i0/i1,j0/j1,k0/k1.
// 2. Calls getColor() on each Coordinate.(Ci*j*k*)
// 3. Computes the return Color Value using a,b,c and the Color values.
template <typename DataT>
DataT ReadPixelDataLinearFiltMode(const cl_int8 CoordValues,
                                  const cl_float4 abc,
                                  const addressing_mode SmplAddrMode,
                                  const range<3> ImgRange, id<3> ImgPitch,
                                  const image_channel_type ImgChannelType,
                                  const image_channel_order ImgChannelOrder,
                                  void *BasePtr, const uint8_t ElementSize) {
  cl_int i0 = CoordValues.s0(), j0 = CoordValues.s1(), k0 = CoordValues.s2(),
         i1 = CoordValues.s4(), j1 = CoordValues.s5(), k1 = CoordValues.s6();

  auto getColorInFloat =
      [&](cl_int4 V) {
        DataT Res = getColor<DataT>(V, SmplAddrMode,
                                    ImgRange, ImgPitch, ImgChannelType,
                                    ImgChannelOrder, BasePtr, ElementSize);
        return Res.template convert<cl_float>();
      };

  // Get Color Values at each Coordinate.
  cl_float4 Ci0j0k0 = getColorInFloat(cl_int4{i0, j0, k0, 0});
  
  cl_float4 Ci1j0k0 = getColorInFloat(cl_int4{i1, j0, k0, 0});
  
  cl_float4 Ci0j1k0 = getColorInFloat(cl_int4{i0, j1, k0, 0});
  
  cl_float4 Ci1j1k0 = getColorInFloat(cl_int4{i1, j1, k0, 0});
  
  cl_float4 Ci0j0k1 = getColorInFloat(cl_int4{i0, j0, k1, 0});
  
  cl_float4 Ci1j0k1 = getColorInFloat(cl_int4{i1, j0, k1, 0});
  
  cl_float4 Ci0j1k1 = getColorInFloat(cl_int4{i0, j1, k1, 0});
  
  cl_float4 Ci1j1k1 = getColorInFloat(cl_int4{i1, j1, k1, 0});

  cl_float a = abc.x();
  cl_float b = abc.y();
  cl_float c = abc.z();

  Ci0j0k0 = (1 - a) * (1 - b) * (1 - c) * Ci0j0k0;
  Ci1j0k0 = a * (1 - b) * (1 - c) * Ci1j0k0;
  Ci0j1k0 = (1 - a) * b * (1 - c) * Ci0j1k0;
  Ci1j1k0 = a * b * (1 - c) * Ci1j1k0;
  Ci0j0k1 = (1 - a) * (1 - b) * c * Ci0j0k1;
  Ci1j0k1 = a * (1 - b) * c * Ci1j0k1;
  Ci0j1k1 = (1 - a) * b * c * Ci0j1k1;
  Ci1j1k1 = a * b * c * Ci1j1k1;

  cl_float4 RetData = Ci0j0k0 + Ci1j0k0 + Ci0j1k0 + Ci1j1k0 + Ci0j0k1 +
                      Ci1j0k1 + Ci0j1k1 + Ci1j1k1;

  // For 2D image:k0 = 0, k1 = 0, c = 0.5
  // RetData = (1 – a) * (1 – b) * Ci0j0 + a * (1 – b) * Ci1j0 +
  //           (1 – a) * b * Ci0j1 + a * b * Ci1j1;
  // For 1D image: j0 = 0, j1 = 0, k0 = 0, k1 = 0, b = 0.5, c = 0.5.
  // RetData = (1 – a) * Ci0 + a * Ci1;
  return RetData.convert<typename TryToGetElementType<DataT>::type>();
}

// imageReadSamplerHostImpl method is called by the read API in image accessors
// for host device.
// Algorithm used: The Algorithm is based on OpenCL spec section 8.2.
// It can be broken down into three major steps:
// Step 1.
//   Check for valid sampler options and Compute u,v,w coordinates:
//   These coordinates are used to compute the Pixel Coordinates that will be
//   read from to compute the return values.
//   u,v,w are normalized for AddrMode:mirror_repeat and repeat.
//   u,v,w are unnormalized for AddrMode:clamp_to_edge, clamp, none.
//       Convert normalized into unnormalized coords using image range.
//   note: When dims=1, u,v,w={u,0,0}
//              dims=2, u,v,w={u,v,0}
//              dims=3, u,v,w-{u,v,w}
// Step 2.
//   Process u,v,w, to find the exact Coordinates to read from:
//   if(Nearest Filtering Mode)
//     compute i,j,k pixel Coordinates based on AddrMode.
//   else(Linear Filtering Mode)
//     compute i0,j0,k0,i1,j1,k1,a,b,c values.
//     Used to load following number of pixels in Step 3.
//       2x2x2 image for Dims=3
//       2x2 image for Dims=2
//       1 pixel for Dims=1 // I think same value should be
//                             returned as nearest case.
// Step 3.
//   Load Image Data, Different for Linear and Nearest Mode:
//     Offset = getOffset based on Coord, ImageRange,ImagePitch.
//   Read values in the appropriate format based on ImgChannelOrder and
//     ImgChannelType.
//   Convert to DataT as per conversion rules in section 8.3 in OpenCL Spec.
//
// TODO: Add additional check for half datatype read.
// Based on OpenCL spec 2.0:
// "The read_imageh calls that take integer coordinates must use a sampler with
// filter mode set to CLK_FILTER_NEAREST, normalized coordinates set to
// CLK_NORMALIZED_COORDS_FALSE and addressing mode set to
// CLK_ADDRESS_CLAMP_TO_EDGE, CLK_ADDRESS_CLAMP or CLK_ADDRESS_NONE; otherwise
// the values returned are undefined."

template <typename CoordT, typename DataT>
DataT imageReadSamplerHostImpl(const CoordT &Coords, const sampler &Smpl,
                               /*All image information*/ range<3> ImgRange,
                               id<3> ImgPitch,
                               image_channel_type ImgChannelType,
                               image_channel_order ImgChannelOrder,
                               void *BasePtr, uint8_t ElementSize) {

  coordinate_normalization_mode SmplNormMode =
      Smpl.get_coordinate_normalization_mode();
  addressing_mode SmplAddrMode = Smpl.get_addressing_mode();
  filtering_mode SmplFiltMode = Smpl.get_filtering_mode();

  CoordT Coorduvw;
  cl_float4 FloatCoorduvw;
  DataT RetData;

  // Step 1:
  // switch-case code is used for a better view on value of Coorduvw for all
  // combinations of Addressing Modes and Normalization Mode.
  switch (SmplNormMode) {
  case coordinate_normalization_mode::unnormalized:
    switch (SmplAddrMode) {
    case addressing_mode::mirrored_repeat:
    case addressing_mode::repeat:
      throw cl::sycl::feature_not_supported(
          "Sampler used with unsupported configuration of "
          "mirrored_repeat/repeat filtering mode with unnormalized "
          "coordinates. ",
          PI_INVALID_OPERATION);
    case addressing_mode::clamp_to_edge:
    case addressing_mode::clamp:
    case addressing_mode::none:
      // Continue with the unnormalized coordinates in Coorduvw.
      Coorduvw = Coords;
      break;
    }
    break; // Break for coordinate_normalization_mode::unnormalized.
  case coordinate_normalization_mode::normalized:
    switch (SmplAddrMode) {
    case addressing_mode::mirrored_repeat:
    case addressing_mode::repeat:
      // Continue with the normalized coordinates in Coorduvw.
      // Based on Section 8.2 Normalised coordinates are used to compute pixel
      // coordinates for addressing_mode::repeat and mirrored_repeat.
      Coorduvw = Coords;
      break;
    case addressing_mode::clamp_to_edge:
    case addressing_mode::clamp:
    case addressing_mode::none:
      // Unnormalize these coordinates.
      // Based on Section 8.2 Normalised coordinats are used to compute pixel
      // coordinates for addressing_mode::clamp/clamp_to_edge and none.
      Coorduvw = UnnormalizeCoordinates(Coords, ImgRange);
      break;
    }
    break; // Break for coordinate_normalization_mode::normalized.
  }

  // Step 2 & Step 3:

  // converToFloat4 converts CoordT of any kind - cl_int, cl_int2, cl_int4,
  // cl_float, cl_float2 and cl_float4 into Coordinates of kind cl_float4 with
  // no loss of precision. For pixel_coordinates already in cl_float4 format,
  // the function returns the same values. This conversion is done to enable
  // implementation of one common function getPixelCoordXXXMode, for any
  // datatype of CoordT passed.
  FloatCoorduvw = convertToFloat4(Coorduvw);
  switch (SmplFiltMode) {
  case filtering_mode::nearest: {
    // Get Pixel Coordinates in integers that will be read from in the Image.
    cl_int4 PixelCoord =
        getPixelCoordNearestFiltMode(FloatCoorduvw, SmplAddrMode, ImgRange);

    // Return Border Color for out-of-range coordinates when Sampler has
    // addressing_mode::clamp. For all other cases and for in-range coordinates
    // read the color and return in DataT type.
    RetData =
        getColor<DataT>(PixelCoord, SmplAddrMode, ImgRange, ImgPitch,
                        ImgChannelType, ImgChannelOrder, BasePtr, ElementSize);
    break;
  }
  case filtering_mode::linear: {
    cl_float4 Retabc;
    // Get Pixel Coordinates in integers that will be read from in the Image.
    // Return i0,j0,k0,0,i1,j1,k1,0 to form 8 coordinates in a 3D image and
    // multiplication factors a,b,c
    cl_int8 CoordValues = getPixelCoordLinearFiltMode(
        FloatCoorduvw, SmplAddrMode, ImgRange, Retabc);

    // Find the 8 coordinates with the values in CoordValues.
    // Computes the Color Value to return.
    RetData = ReadPixelDataLinearFiltMode<DataT>(
        CoordValues, Retabc, SmplAddrMode, ImgRange, ImgPitch, ImgChannelType,
        ImgChannelOrder, BasePtr, ElementSize);

    break;
  }
  }

  return RetData;
}

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
#endif
