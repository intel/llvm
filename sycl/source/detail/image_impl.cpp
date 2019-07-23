//==------------ image_impl.cpp --------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/image.hpp>

namespace cl {
namespace sycl {
namespace detail {

uint8_t getImageNumberChannels(image_channel_order Order) {
  switch (Order) {
  case image_channel_order::a:
  case image_channel_order::r:
  case image_channel_order::rx:
  case image_channel_order::intensity:
  case image_channel_order::luminance:
    return 1;
  case image_channel_order::rg:
  case image_channel_order::rgx:
  case image_channel_order::ra:
    return 2;
  case image_channel_order::rgb:
  case image_channel_order::rgbx:
    return 3;
  case image_channel_order::rgba:
  case image_channel_order::argb:
  case image_channel_order::bgra:
  case image_channel_order::abgr:
    return 4;
  }
  assert(!"Unhandled image channel order");
  return 0;
}

// Returns the number of bytes per image element
uint8_t getImageElementSize(uint8_t NumChannels, image_channel_type Type) {
  size_t Retval = 0;
  switch (Type) {
  case image_channel_type::snorm_int8:
  case image_channel_type::unorm_int8:
  case image_channel_type::signed_int8:
  case image_channel_type::unsigned_int8:
    Retval = NumChannels;
    break;
  case image_channel_type::snorm_int16:
  case image_channel_type::unorm_int16:
  case image_channel_type::signed_int16:
  case image_channel_type::unsigned_int16:
  case image_channel_type::fp16:
    Retval = 2 * NumChannels;
    break;
  case image_channel_type::signed_int32:
  case image_channel_type::unsigned_int32:
  case image_channel_type::fp32:
    Retval = 4 * NumChannels;
    break;
  case image_channel_type::unorm_short_565:
  case image_channel_type::unorm_short_555:
    Retval = 2;
    break;
  case image_channel_type::unorm_int_101010:
    Retval = 4;
    break;
  default:
    assert(!"Unhandled image channel type");
  }
  // OpenCL states that "The number of bits per element determined by the
  // image_channel_type and image_channel_order must be a power of two"
  // Retval is in bytes. The formula remains the same for bytes or bits.
  assert(((Retval - 1) & Retval) == 0);
  return Retval;
}

RT::PiMemImageChannelOrder convertChannelOrder(image_channel_order Order) {
  switch (Order) {
  case image_channel_order::a:
    return PI_IMAGE_CHANNEL_ORDER_A;
  case image_channel_order::r:
    return PI_IMAGE_CHANNEL_ORDER_R;
  case image_channel_order::rx:
    return PI_IMAGE_CHANNEL_ORDER_Rx;
  case image_channel_order::rg:
    return PI_IMAGE_CHANNEL_ORDER_RG;
  case image_channel_order::rgx:
    return PI_IMAGE_CHANNEL_ORDER_RGx;
  case image_channel_order::ra:
    return PI_IMAGE_CHANNEL_ORDER_RA;
  case image_channel_order::rgb:
    return PI_IMAGE_CHANNEL_ORDER_RGB;
  case image_channel_order::rgbx:
    return PI_IMAGE_CHANNEL_ORDER_RGBx;
  case image_channel_order::rgba:
    return PI_IMAGE_CHANNEL_ORDER_RGBA;
  case image_channel_order::argb:
    return PI_IMAGE_CHANNEL_ORDER_ARGB;
  case image_channel_order::bgra:
    return PI_IMAGE_CHANNEL_ORDER_BGRA;
  case image_channel_order::intensity:
    return PI_IMAGE_CHANNEL_ORDER_INTENSITY;
  case image_channel_order::luminance:
    return PI_IMAGE_CHANNEL_ORDER_LUMINANCE;
  case image_channel_order::abgr:
    return PI_IMAGE_CHANNEL_ORDER_ABGR;
  default: {
    assert(!"Unhandled image_channel_order");
    return static_cast<RT::PiMemImageChannelOrder>(0);
  }
  }
}

image_channel_order convertChannelOrder(RT::PiMemImageChannelOrder Order) {
  switch (Order) {
  case PI_IMAGE_CHANNEL_ORDER_A:
    return image_channel_order::a;
  case PI_IMAGE_CHANNEL_ORDER_R:
    return image_channel_order::r;
  case PI_IMAGE_CHANNEL_ORDER_Rx:
    return image_channel_order::rx;
  case PI_IMAGE_CHANNEL_ORDER_RG:
    return image_channel_order::rg;
  case PI_IMAGE_CHANNEL_ORDER_RGx:
    return image_channel_order::rgx;
  case PI_IMAGE_CHANNEL_ORDER_RA:
    return image_channel_order::ra;
  case PI_IMAGE_CHANNEL_ORDER_RGB:
    return image_channel_order::rgb;
  case PI_IMAGE_CHANNEL_ORDER_RGBx:
    return image_channel_order::rgbx;
  case PI_IMAGE_CHANNEL_ORDER_RGBA:
    return image_channel_order::rgba;
  case PI_IMAGE_CHANNEL_ORDER_ARGB:
    return image_channel_order::argb;
  case PI_IMAGE_CHANNEL_ORDER_BGRA:
    return image_channel_order::bgra;
  case PI_IMAGE_CHANNEL_ORDER_INTENSITY:
    return image_channel_order::intensity;
  case PI_IMAGE_CHANNEL_ORDER_LUMINANCE:
    return image_channel_order::luminance;
  case PI_IMAGE_CHANNEL_ORDER_ABGR:
    return image_channel_order::abgr;
  default: {
    assert(!"Unhandled image_channel_order");
    return static_cast<image_channel_order>(0);
  }
  }
}

RT::PiMemImageChannelType convertChannelType(image_channel_type Type) {
  switch (Type) {
  case image_channel_type::snorm_int8:
    return PI_IMAGE_CHANNEL_TYPE_SNORM_INT8;
  case image_channel_type::snorm_int16:
    return PI_IMAGE_CHANNEL_TYPE_SNORM_INT16;
  case image_channel_type::unorm_int8:
    return PI_IMAGE_CHANNEL_TYPE_UNORM_INT8;
  case image_channel_type::unorm_int16:
    return PI_IMAGE_CHANNEL_TYPE_UNORM_INT16;
  case image_channel_type::unorm_short_565:
    return PI_IMAGE_CHANNEL_TYPE_UNORM_SHORT_565;
  case image_channel_type::unorm_short_555:
    return PI_IMAGE_CHANNEL_TYPE_UNORM_SHORT_555;
  case image_channel_type::unorm_int_101010:
    return PI_IMAGE_CHANNEL_TYPE_UNORM_INT_101010;
  case image_channel_type::signed_int8:
    return PI_IMAGE_CHANNEL_TYPE_SIGNED_INT8;
  case image_channel_type::signed_int16:
    return PI_IMAGE_CHANNEL_TYPE_SIGNED_INT16;
  case image_channel_type::signed_int32:
    return PI_IMAGE_CHANNEL_TYPE_SIGNED_INT32;
  case image_channel_type::unsigned_int8:
    return PI_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8;
  case image_channel_type::unsigned_int16:
    return PI_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16;
  case image_channel_type::unsigned_int32:
    return PI_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32;
  case image_channel_type::fp16:
    return PI_IMAGE_CHANNEL_TYPE_HALF_FLOAT;
  case image_channel_type::fp32:
    return PI_IMAGE_CHANNEL_TYPE_FLOAT;
  default: {
    assert(!"Unhandled image_channel_order");
    return static_cast<RT::PiMemImageChannelType>(0);
  }
  }
}

image_channel_type convertChannelType(RT::PiMemImageChannelType Type) {
  switch (Type) {
  case PI_IMAGE_CHANNEL_TYPE_SNORM_INT8:
    return image_channel_type::snorm_int8;
  case PI_IMAGE_CHANNEL_TYPE_SNORM_INT16:
    return image_channel_type::snorm_int16;
  case PI_IMAGE_CHANNEL_TYPE_UNORM_INT8:
    return image_channel_type::unorm_int8;
  case PI_IMAGE_CHANNEL_TYPE_UNORM_INT16:
    return image_channel_type::unorm_int16;
  case PI_IMAGE_CHANNEL_TYPE_UNORM_SHORT_565:
    return image_channel_type::unorm_short_565;
  case PI_IMAGE_CHANNEL_TYPE_UNORM_SHORT_555:
    return image_channel_type::unorm_short_555;
  case PI_IMAGE_CHANNEL_TYPE_UNORM_INT_101010:
    return image_channel_type::unorm_int_101010;
  case PI_IMAGE_CHANNEL_TYPE_SIGNED_INT8:
    return image_channel_type::signed_int8;
  case PI_IMAGE_CHANNEL_TYPE_SIGNED_INT16:
    return image_channel_type::signed_int16;
  case PI_IMAGE_CHANNEL_TYPE_SIGNED_INT32:
    return image_channel_type::signed_int32;
  case PI_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8:
    return image_channel_type::unsigned_int8;
  case PI_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16:
    return image_channel_type::unsigned_int16;
  case PI_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32:
    return image_channel_type::unsigned_int32;
  case PI_IMAGE_CHANNEL_TYPE_HALF_FLOAT:
    return image_channel_type::fp16;
  case PI_IMAGE_CHANNEL_TYPE_FLOAT:
    return image_channel_type::fp32;
  default: {
    assert(!"Unhandled image_channel_order");
    return static_cast<image_channel_type>(0);
  }
  }
}

} // namespace detail
} // namespace sycl
} // namespace cl
