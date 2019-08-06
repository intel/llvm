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
#ifndef __SYCL_DEVICE_ONLY__
#include <CL/sycl/detail/generic_type_traits.hpp>
#include <CL/sycl/image.hpp>
#include <CL/sycl/sampler.hpp>
#include <CL/sycl/types.hpp>

namespace cl {
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
// Non-valid coordinates are written as 0.
template <typename T>
detail::enable_if_t<IsValidCoordType<T>::value, cl_float4>
convertToFloat4(T Coords) {
  return {static_cast<float>(Coords), 0.f, 0.f, 0.f};
}

template <typename T>
detail::enable_if_t<IsValidCoordType<T>::value, cl_float4>
convertToFloat4(vec<T, 2> Coords) {
  return {static_cast<float>(Coords.x()), static_cast<float>(Coords.y()), 0.f,
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
getImageOffset(const T &Coords, id<3> ImgPitch, const uint8_t ElementSize) {
  return Coords * ElementSize;
}

template <typename T>
detail::enable_if_t<std::is_integral<T>::value, size_t>
getImageOffset(const vec<T, 2> &Coords, id<3> ImgPitch,
               const uint8_t ElementSize) {
  return Coords.x() * ElementSize + Coords.y() * ImgPitch[0];
}

template <typename T>
detail::enable_if_t<std::is_integral<T>::value, size_t>
getImageOffset(const vec<T, 4> &Coords, id<3> ImgPitch,
               const uint8_t ElementSize) {
  return Coords.x() * ElementSize + Coords.y() * ImgPitch[0] +
         Coords.z() * ImgPitch[1];
}

// Process cl_float4 Coordinates and return the appropriate Pixel Coordinates to
// read from based on Addressing Mode for Nearest filter mode.
cl_int4 getPixelCoordNearestFiltMode(cl_float4, addressing_mode, range<3>);

// Reads data from a pixel at Ptr location, based on the number of Channels in
// Order and returns the data.
// The datatype used to read from the Ptr is based on the T of the
// image. This datatype is computed by the calling API.
template <typename T> vec<T, 4> readPixel(T *Ptr, image_channel_order Order) {
  vec<T, 4> Pixel;
  const uint8_t NumChannels = getImageNumberChannels(Order);

  switch (NumChannels) {
  case 4:
    Pixel.w() = Ptr[3];
  case 3:
    Pixel.z() = Ptr[2];
  case 2:
    Pixel.y() = Ptr[1];
  case 1:
    Pixel.x() = Ptr[0];
    break;
  default:
    assert(!"Unhandled image channel order");
    break;
  }
  return Pixel;
}

// Write data to a pixel at Ptr location, based on the number of Channels in
// ImageChannelOrder. The data passed to this API in 'Pixel' is already
// converted to Datatype of the Channel based on ImageChannelType by the calling
// API.
template <typename T>
void writePixel(vec<T, 4> Pixel, T *Ptr, image_channel_order Order) {
  const uint8_t NumChannels = getImageNumberChannels(Order);

  switch (NumChannels) {
  case 4:
    Ptr[3] = Pixel.w();
  case 3:
    Ptr[2] = Pixel.z();
  case 2:
    Ptr[1] = Pixel.y();
  case 1:
    Ptr[0] = Pixel.x();
    break;
  default:
    assert(!"Unhandled image channel order");
    break;
  }
}

// Converts read pixel data into return datatype based on the channel type of
// the image.
// TODO: Change this method to use the conversion rules as given in the OpenCL
// Spec section 8.3. The conversion rules may be handled differently for each
// return datatype - float, int32, uint32, half. ImageChannelType is passed to
// the function to use appropriate conversion rules.
template <typename ChannelType, typename RetDataType,
          typename = detail::enable_if_t<
              (detail::is_contained<
                  RetDataType, type_list<cl_int, cl_float, cl_uint>>::value)>>
void convertReadData(vec<ChannelType, 4> PixelData,
                     image_channel_type ImageChannelType,
                     vec<RetDataType, 4> &RetData) {
  RetData.x() = (RetDataType)PixelData.x();
  RetData.y() = (RetDataType)PixelData.y();
  RetData.z() = (RetDataType)PixelData.z();
  RetData.w() = (RetDataType)PixelData.w();
}

// Separate function for half datatype.
// Float is used as typecast to resolve ambiguity when converting data into half
// datatype.
template <typename ChannelType>
void convertReadData(vec<ChannelType, 4> PixelData,
                     image_channel_type ImageChannelType,
                     vec<cl_half, 4> &RetData) {
  RetData.x() = (float)PixelData.x();
  RetData.y() = (float)PixelData.y();
  RetData.z() = (float)PixelData.z();
  RetData.w() = (float)PixelData.w();
}

// Converts data to write into appropriate datatype based on the channel of the
// image.
// TODO: Change this method to use the conversion rules as given in the OpenCL
// Spec Section 8.3. The conversion rules may be handled differently for each
// return datatype - float, int32, uint32, half. ImageChannelType is passed to
// the function to use appropriate conversion rules.
template <typename ChannelType, typename WriteDataType>
vec<ChannelType, 4> convertWriteData(vec<WriteDataType, 4> WriteData,
                                     image_channel_type ImageChannelType) {
  vec<ChannelType, 4> PixelData;
  PixelData.x() = (ChannelType)WriteData.x();
  PixelData.y() = (ChannelType)WriteData.y();
  PixelData.z() = (ChannelType)WriteData.z();
  PixelData.w() = (ChannelType)WriteData.w();
  return PixelData;
}

// Separate function for half datatype.
// To resolve ambiguity when converting data into half datatype,float is used as
// typecast.
template <typename WriteDataType>
vec<cl_half, 4> convertWriteDataToHalf(vec<WriteDataType, 4> WriteData,
                                       image_channel_type ImageChannelType) {
  vec<cl_half, 4> PixelData;
  PixelData.x() = (float)WriteData.x();
  PixelData.y() = (float)WriteData.y();
  PixelData.z() = (float)WriteData.z();
  PixelData.w() = (float)WriteData.w();
  return PixelData;
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
    writePixel(convertWriteData<int8_t,
                                typename TryToGetElementType<WriteDataT>::type>(
                   Color, ImgChannelType),
               reinterpret_cast<int8_t *>(Ptr), ImgChannelOrder);
    break;
  case image_channel_type::snorm_int16:
    writePixel(convertWriteData<int16_t,
                                typename TryToGetElementType<WriteDataT>::type>(
                   Color, ImgChannelType),
               reinterpret_cast<int16_t *>(Ptr), ImgChannelOrder);
    break;
  case image_channel_type::unorm_int8:
    writePixel(convertWriteData<uint8_t,
                                typename TryToGetElementType<WriteDataT>::type>(
                   Color, ImgChannelType),
               reinterpret_cast<uint8_t *>(Ptr), ImgChannelOrder);
    break;
  case image_channel_type::unorm_int16:
    writePixel(convertWriteData<uint16_t,
                                typename TryToGetElementType<WriteDataT>::type>(
                   Color, ImgChannelType),
               reinterpret_cast<uint16_t *>(Ptr), ImgChannelOrder);
    break;
  case image_channel_type::unorm_short_565:
    writePixel(
        convertWriteData<short, typename TryToGetElementType<WriteDataT>::type>(
            Color, ImgChannelType),
        reinterpret_cast<short *>(Ptr), ImgChannelOrder);
    break;
  case image_channel_type::unorm_short_555:
    writePixel(
        convertWriteData<short, typename TryToGetElementType<WriteDataT>::type>(
            Color, ImgChannelType),
        reinterpret_cast<short *>(Ptr), ImgChannelOrder);
    break;
  case image_channel_type::unorm_int_101010:
    writePixel(convertWriteData<uint32_t,
                                typename TryToGetElementType<WriteDataT>::type>(
                   Color, ImgChannelType),
               reinterpret_cast<uint32_t *>(Ptr), ImgChannelOrder);
    break;
  case image_channel_type::signed_int8:
    writePixel(convertWriteData<int8_t,
                                typename TryToGetElementType<WriteDataT>::type>(
                   Color, ImgChannelType),
               reinterpret_cast<int8_t *>(Ptr), ImgChannelOrder);
    break;
  case image_channel_type::signed_int16:
    writePixel(convertWriteData<int16_t,
                                typename TryToGetElementType<WriteDataT>::type>(
                   Color, ImgChannelType),
               reinterpret_cast<int16_t *>(Ptr), ImgChannelOrder);
    break;
  case image_channel_type::signed_int32:
    writePixel(convertWriteData<int32_t,
                                typename TryToGetElementType<WriteDataT>::type>(
                   Color, ImgChannelType),
               reinterpret_cast<int32_t *>(Ptr), ImgChannelOrder);
    break;
  case image_channel_type::unsigned_int8:
    writePixel(convertWriteData<uint8_t,
                                typename TryToGetElementType<WriteDataT>::type>(
                   Color, ImgChannelType),
               reinterpret_cast<uint8_t *>(Ptr), ImgChannelOrder);
    break;
  case image_channel_type::unsigned_int16:
    writePixel(convertWriteData<uint16_t,
                                typename TryToGetElementType<WriteDataT>::type>(
                   Color, ImgChannelType),
               reinterpret_cast<uint16_t *>(Ptr), ImgChannelOrder);
    break;
  case image_channel_type::unsigned_int32:
    writePixel(convertWriteData<uint32_t,
                                typename TryToGetElementType<WriteDataT>::type>(
                   Color, ImgChannelType),
               reinterpret_cast<uint32_t *>(Ptr), ImgChannelOrder);
    break;
  case image_channel_type::fp16:
    writePixel(
        convertWriteDataToHalf<typename TryToGetElementType<WriteDataT>::type>(
            Color, ImgChannelType),
        reinterpret_cast<cl_half *>(Ptr), ImgChannelOrder);
    break;
  case image_channel_type::fp32:
    writePixel(
        convertWriteData<float, typename TryToGetElementType<WriteDataT>::type>(
            Color, ImgChannelType),
        reinterpret_cast<float *>(Ptr), ImgChannelOrder);
    break;
  }
}

// Method called to read a Coord by imageReadSamplerHostImpl when filter mode in
// the Sampler is Nearest. This method takes Unnormalized Coords - 'PixelCoord'
// as cl_int4. Invalid Coord are denoted by 0. Steps:
// 1. Compute Offset for given Unnormalised Coordinates using ImagePitch and
// ElementSize.(getImageOffset)
// 2. Add this Offset to BasePtr to compute the location of the Image.
// 3. Convert this Ptr to the appropriate datatype pointer based on
// ImageChannelType. (reinterpret_cast)
// 4. Read the appropriate number of channels(computed using
// ImageChannelOrder) of the appropriate Channel datatype into Color
// variable.(readPixel)
// 5. Convert the Read Data into Return DataTy based on conversion rules in
// the Spec.(convertReadData)
// Possible DataTy are cl_int4, cl_uint4, cl_float4, cl_half4;
template <typename DataTy>
DataTy ReadPixelDataNearestFiltMode(cl_int4 PixelCoord, id<3> ImgPitch,
                                    image_channel_type ImageChannelType,
                                    image_channel_order ImageChannelOrder,
                                    void *BasePtr, uint8_t ElementSize) {
  DataTy Color = {0, 0, 0, 0};
  auto Ptr = static_cast<unsigned char *>(BasePtr) +
             getImageOffset(PixelCoord, ImgPitch,
                            ElementSize); // Utility to compute offset in
                                          // image_accessor_util.hpp

  switch (ImageChannelType) {
  case image_channel_type::snorm_int8:
    convertReadData<int8_t>(
        readPixel(reinterpret_cast<int8_t *>(Ptr), ImageChannelOrder),
        image_channel_type::snorm_int8, Color);
    break;
  case image_channel_type::snorm_int16:
    convertReadData<int16_t>(
        readPixel(reinterpret_cast<int16_t *>(Ptr), ImageChannelOrder),
        image_channel_type::snorm_int16, Color);
    break;
  case image_channel_type::unorm_int8:
    convertReadData<uint8_t>(
        readPixel(reinterpret_cast<uint8_t *>(Ptr), ImageChannelOrder),
        image_channel_type::unorm_int8, Color);
    break;
  case image_channel_type::unorm_int16:
    convertReadData<uint16_t>(
        readPixel(reinterpret_cast<uint16_t *>(Ptr), ImageChannelOrder),
        image_channel_type::unorm_int16, Color);
    break;
  case image_channel_type::unorm_short_565:
    convertReadData<ushort>(
        readPixel(reinterpret_cast<ushort *>(Ptr), ImageChannelOrder),
        image_channel_type::unorm_short_565, Color);
    break;
  case image_channel_type::unorm_short_555:
    convertReadData<ushort>(
        readPixel(reinterpret_cast<ushort *>(Ptr), ImageChannelOrder),
        image_channel_type::unorm_short_555, Color);
    break;
  case image_channel_type::unorm_int_101010:
    convertReadData<uint32_t>(
        readPixel(reinterpret_cast<uint32_t *>(Ptr), ImageChannelOrder),
        image_channel_type::unorm_int_101010, Color);
    break;
  case image_channel_type::signed_int8:
    convertReadData<int8_t>(
        readPixel(reinterpret_cast<int8_t *>(Ptr), ImageChannelOrder),
        image_channel_type::signed_int8, Color);
    break;
  case image_channel_type::signed_int16:
    convertReadData<int16_t>(
        readPixel(reinterpret_cast<int16_t *>(Ptr), ImageChannelOrder),
        image_channel_type::signed_int16, Color);
    break;
  case image_channel_type::signed_int32:
    convertReadData<int32_t>(
        readPixel(reinterpret_cast<int32_t *>(Ptr), ImageChannelOrder),
        image_channel_type::signed_int32, Color);
    break;
  case image_channel_type::unsigned_int8:
    convertReadData<uint8_t>(
        readPixel(reinterpret_cast<uint8_t *>(Ptr), ImageChannelOrder),
        image_channel_type::unsigned_int8, Color);
    break;
  case image_channel_type::unsigned_int16:
    convertReadData<uint16_t>(
        readPixel(reinterpret_cast<uint16_t *>(Ptr), ImageChannelOrder),
        image_channel_type::unsigned_int16, Color);
    break;
  case image_channel_type::unsigned_int32:
    convertReadData<uint32_t>(
        readPixel(reinterpret_cast<uint32_t *>(Ptr), ImageChannelOrder),
        image_channel_type::unsigned_int32, Color);
    break;
  case image_channel_type::fp16:
    convertReadData<cl_half>(
        readPixel(reinterpret_cast<cl_half *>(Ptr), ImageChannelOrder),
        image_channel_type::fp16, Color);
    break;
  case image_channel_type::fp32:
    convertReadData<float>(
        readPixel(reinterpret_cast<float *>(Ptr), ImageChannelOrder),
        image_channel_type::fp32, Color);
    break;
  }

  return Color;
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
// TODO:
// Extend support for Step2 and Step3 for Linear Filtering Mode.
// Extend support to find out of bounds Coordinates and return appropriate
// value based on Addressing Mode.

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
          "coordinates. ");
      break;
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
  cl_int4 PixelCoord;
  switch (SmplFiltMode) {
  case filtering_mode::nearest: {
    // Get Pixel Coordinates in integers that will be read from in the Image.
    PixelCoord =
        getPixelCoordNearestFiltMode(FloatCoorduvw, SmplAddrMode, ImgRange);
    // TODO: Check Out-of-range coordinates. Need to use Addressing Mode Of
    // Sampler to find the appropriate return value. Eg: clamp_to_edge returns
    // edge values and clamp returns border color for out-of-range coordinates.
    RetData = ReadPixelDataNearestFiltMode<DataT>(
        PixelCoord, ImgPitch, ImgChannelType, ImgChannelOrder, BasePtr,
        ElementSize);
    break;
  }
  case filtering_mode::linear:
    // TO DO: To implement this function.
    // cl_float3 ValueAbc = getPixelCoordsLinearFiltMode(
    //     FloatCoorduvw, SmplAddrMode,
    //    ImgRange, cl_int4 i0j0k0, cl_int4 i1j1k1); // Get total 9 variables.
    // LoadData based on 9 returned values.
    // Ret_Data = ReadPixelDataLinearFiltMode(...);
    break;
  }

  return RetData;
}

} // namespace detail
} // namespace sycl
} // namespace cl
#endif
