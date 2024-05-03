//==----------- image_accessor_util.cpp ------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/accessor.hpp>
#include <sycl/accessor_image.hpp>
#include <sycl/builtins.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {

// For Nearest Filtering mode, process float4 Coordinates and
// return the appropriate Pixel Coordinates based on Addressing Mode.
int4 getPixelCoordNearestFiltMode(float4 Coorduvw,
                                  const addressing_mode SmplAddrMode,
                                  const range<3> ImgRange) {
  int4 Coordijk(0);
  int4 Rangewhd(ImgRange[0], ImgRange[1], ImgRange[2], 0);
  switch (SmplAddrMode) {
  case addressing_mode::mirrored_repeat: {
    float4 Tempuvw(0);
    Tempuvw = 2.0f * sycl::rint(0.5f * Coorduvw);
    Tempuvw = sycl::fabs(Coorduvw - Tempuvw);
    Tempuvw = Tempuvw * (Rangewhd.convert<cl_float>());
    Tempuvw = (sycl::floor(Tempuvw));
    Coordijk = Tempuvw.convert<cl_int>();
    Coordijk = sycl::min(Coordijk, (Rangewhd - 1));
    // Eg:
    // u,v,w = {2.3,1.7,0.5} // normalized coordinates.
    // w,h,d = {9,9,9}
    // u1=2*rint(1.15)=2
    // v1=2*rint(0.85)=2
    // w1=2*rint(0.5)=0
    // u1=fabs(2.3-2)=.3
    // v1=fabs(1.7-2)=.3
    // w1=fabs(0.5-0)=.5
    // u1=0.3*9=2.7
    // v1=0.3*9=2.7
    // w1=0.5*9=4.5
    // i,j,k = {2,2,4}

  } break;
  case addressing_mode::repeat: {

    float4 Tempuvw(0);
    Tempuvw = (Coorduvw - sycl::floor(Coorduvw)) * Rangewhd.convert<cl_float>();
    Coordijk = (sycl::floor(Tempuvw)).convert<cl_int>();
    int4 GreaterThanEqual = (Coordijk >= Rangewhd);
    Coordijk = sycl::select(Coordijk, (Coordijk - Rangewhd), GreaterThanEqual);
    // Eg:
    // u = 2.3; v = 1.5; w = 0.5; // normalized coordinates.
    // w,h,d  = {9,9,9};
    // u1= 0.3*w;
    // v1= 0.5*d;
    // w1= 0.5*h;
    // i = floor(2.7);
    // j = floor(4.5);
    // k = floor(4.5);
    // if (i/j/k > w/h/d-1)
    //      // Condition is not satisfied.
    //      (This condition I think will only be satisfied if the floating point
    //      arithmetic of  multiplication
    //      gave a value in u1/v1/w1 as > w/h/d)
    // i = 2; j = 4; k = 4;
  } break;
  case addressing_mode::clamp_to_edge:
    Coordijk = (sycl::floor(Coorduvw)).convert<cl_int>();
    Coordijk = sycl::clamp(Coordijk, int4(0), (Rangewhd - 1));
    break;
  case addressing_mode::clamp:
    Coordijk = (sycl::floor(Coorduvw)).convert<cl_int>();
    Coordijk = sycl::clamp(Coordijk, int4(-1), Rangewhd);
    break;
  case addressing_mode::none:
    Coordijk = (sycl::floor(Coorduvw)).convert<cl_int>();
    break;
  }
  return Coordijk;
}

// For Linear Filtering mode, process Coordinates-Coord_uvw and return
// coordinate indexes based on Addressing Mode.
// The value returned contains (i0,j0,k0,0,i1,j1,k1,0).
// Retabc contains the values of (a,b,c,0)
// The caller of this function should use these values to create the 8 pixel
// coordinates and multiplication coefficients.
int8 getPixelCoordLinearFiltMode(float4 Coorduvw,
                                 const addressing_mode SmplAddrMode,
                                 const range<3> ImgRange, float4 &Retabc) {
  int4 Rangewhd(ImgRange[0], ImgRange[1], ImgRange[2], 0);
  int4 Ci0j0k0(0);
  int4 Ci1j1k1(0);
  int4 Int_uvwsubhalf = sycl::floor(Coorduvw - 0.5f).convert<cl_int>();

  switch (SmplAddrMode) {
  case addressing_mode::mirrored_repeat: {
    float4 Temp;
    Temp = (sycl::rint(Coorduvw * 0.5f)) * 2.0f;
    Temp = sycl::fabs(Coorduvw - Temp);
    Coorduvw = Temp * Rangewhd.convert<cl_float>();
    Int_uvwsubhalf = sycl::floor(Coorduvw - 0.5f).convert<cl_int>();

    Ci0j0k0 = Int_uvwsubhalf;
    Ci1j1k1 = Ci0j0k0 + 1;

    Ci0j0k0 = sycl::max(Ci0j0k0, 0);
    Ci1j1k1 = sycl::min(Ci1j1k1, (Rangewhd - 1));
  } break;
  case addressing_mode::repeat: {

    Coorduvw =
        (Coorduvw - sycl::floor(Coorduvw)) * Rangewhd.convert<cl_float>();
    Int_uvwsubhalf = sycl::floor(Coorduvw - 0.5f).convert<cl_int>();

    Ci0j0k0 = Int_uvwsubhalf;
    Ci1j1k1 = Ci0j0k0 + 1;

    Ci0j0k0 = sycl::select(Ci0j0k0, (Ci0j0k0 + Rangewhd), Ci0j0k0 < int4(0));
    Ci1j1k1 = sycl::select(Ci1j1k1, (Ci1j1k1 - Rangewhd), Ci1j1k1 >= Rangewhd);

  } break;
  case addressing_mode::clamp_to_edge: {
    Ci0j0k0 = sycl::clamp(Int_uvwsubhalf, int4(0), (Rangewhd - 1));
    Ci1j1k1 = sycl::clamp((Int_uvwsubhalf + 1), int4(0), (Rangewhd - 1));
    break;
  }
  case addressing_mode::clamp: {
    Ci0j0k0 = sycl::clamp(Int_uvwsubhalf, int4(-1), Rangewhd);
    Ci1j1k1 = sycl::clamp((Int_uvwsubhalf + 1), int4(-1), Rangewhd);
    break;
  }
  case addressing_mode::none: {
    Ci0j0k0 = Int_uvwsubhalf;
    Ci1j1k1 = Ci0j0k0 + 1;
    break;
  }
  }
  Retabc = (Coorduvw - 0.5f) - (Int_uvwsubhalf.convert<cl_float>());
  Retabc.w() = 0.0f;
  return int8(Ci0j0k0, Ci1j1k1);
}

// Function returns true when PixelCoord is out of image's range.
// It is only valid for addressing_mode::clamp/none.
// Note: For addressing_mode::none , spec says outofrange access is not defined.
// This function handles this addressing_mode to avoid accessing out of bound
// memories on host.
bool isOutOfRange(const int4 PixelCoord, const addressing_mode SmplAddrMode,
                  const range<3> ImgRange) {

  if (SmplAddrMode != addressing_mode::clamp &&
      SmplAddrMode != addressing_mode::none)
    return false;

  auto CheckOutOfRange = [](cl_int Coord, cl_int Range) {
    return ((Coord < 0) || (Coord >= Range));
  };

  bool CheckWidth = CheckOutOfRange(PixelCoord.x(), ImgRange[0]);
  bool CheckHeight = CheckOutOfRange(PixelCoord.y(), ImgRange[1]);
  bool CheckDepth = CheckOutOfRange(PixelCoord.z(), ImgRange[2]);

  return (CheckWidth || CheckHeight || CheckDepth);
}

float4 getBorderColor(const image_channel_order ImgChannelOrder) {

  float4 BorderColor(0.0f);
  switch (ImgChannelOrder) {
  case image_channel_order::r:
  case image_channel_order::rg:
  case image_channel_order::rgb:
  case image_channel_order::luminance:
    BorderColor.w() = 1.0f;
    break;
  default:
    break;
  }
  return BorderColor;
}

} // namespace detail
} // namespace _V1
} // namespace sycl
