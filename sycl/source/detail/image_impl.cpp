//==------------ image_impl.cpp --------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/detail/image_impl.hpp>
#include <CL/sycl/detail/memory_manager.hpp>
#include <CL/sycl/image.hpp>
#include <detail/context_impl.hpp>

#include <algorithm>
#include <vector>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

template <info::device Param>
static bool checkImageValueRange(const std::vector<device> &Devices,
                                 const size_t Value) {
  return Value >= 1 && std::all_of(Devices.cbegin(), Devices.cend(),
                                   [Value](const device &Dev) {
                                     return Value <= Dev.get_info<Param>();
                                   });
}

template <typename T, typename... Args> static bool checkAnyImpl(T) {
  return false;
}

template <typename ValT, typename VarT, typename... Args>
static bool checkAnyImpl(ValT Value, VarT Variant, Args... Arguments) {
  return (Value == Variant) ? true : checkAnyImpl(Value, Arguments...);
}

template <typename T, typename... Args>
static bool checkAny(const T Value, Args... Arguments) {
  return checkAnyImpl(Value, Arguments...);
}

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
    return 3;
  case image_channel_order::rgbx:
  case image_channel_order::rgba:
  case image_channel_order::argb:
  case image_channel_order::bgra:
  case image_channel_order::abgr:
  case image_channel_order::ext_oneapi_srgba:
    return 4;
  }
  assert(false && "Unhandled image channel order");
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
  case image_channel_order::ext_oneapi_srgba:
    return PI_IMAGE_CHANNEL_ORDER_sRGBA;
  }
  assert(false && "Unhandled image_channel_order");
  return static_cast<RT::PiMemImageChannelOrder>(0);
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
  case PI_IMAGE_CHANNEL_ORDER_sRGBA:
    return image_channel_order::ext_oneapi_srgba;
  }
  assert(false && "Unhandled image_channel_order");
  return static_cast<image_channel_order>(0);
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
  }
  assert(false && "Unhandled image_channel_order");
  return static_cast<RT::PiMemImageChannelType>(0);
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
  }
  assert(false && "Unhandled image_channel_order");
  return static_cast<image_channel_type>(0);
}

template <typename T>
static void getImageInfo(const ContextImplPtr Context, RT::PiMemImageInfo Info,
                         T &Dest, RT::PiMem InteropMemObject) {
  const detail::plugin &Plugin = Context->getPlugin();
  RT::PiMem Mem = pi::cast<RT::PiMem>(InteropMemObject);
  Plugin.call<PiApiKind::piMemImageGetInfo>(Mem, Info, sizeof(T), &Dest,
                                            nullptr);
}

template <int Dimensions>
image_impl<Dimensions>::image_impl(
    cl_mem MemObject, const context &SyclContext, event AvailableEvent,
    std::unique_ptr<SYCLMemObjAllocator> Allocator)
    : BaseT(MemObject, SyclContext, std::move(AvailableEvent),
            std::move(Allocator)),
      MRange(InitializedVal<Dimensions, range>::template get<0>()) {
  RT::PiMem Mem = pi::cast<RT::PiMem>(BaseT::MInteropMemObject);
  const ContextImplPtr Context = getSyclObjImpl(SyclContext);
  const detail::plugin &Plugin = Context->getPlugin();
  Plugin.call<PiApiKind::piMemGetInfo>(Mem, CL_MEM_SIZE, sizeof(size_t),
                                       &(BaseT::MSizeInBytes), nullptr);

  RT::PiMemImageFormat Format;
  getImageInfo(Context, PI_IMAGE_INFO_FORMAT, Format, Mem);
  MOrder = detail::convertChannelOrder(Format.image_channel_order);
  MType = detail::convertChannelType(Format.image_channel_data_type);
  MNumChannels = getImageNumberChannels(MOrder);

  getImageInfo(Context, PI_IMAGE_INFO_ELEMENT_SIZE, MElementSize, Mem);
  assert(getImageElementSize(MNumChannels, MType) == MElementSize);

  getImageInfo(Context, PI_IMAGE_INFO_ROW_PITCH, MRowPitch, Mem);
  getImageInfo(Context, PI_IMAGE_INFO_SLICE_PITCH, MSlicePitch, Mem);

  switch (Dimensions) {
  case 3:
    getImageInfo(Context, PI_IMAGE_INFO_DEPTH, MRange[2], Mem);
    __SYCL_FALLTHROUGH;
  case 2:
    getImageInfo(Context, PI_IMAGE_INFO_HEIGHT, MRange[1], Mem);
    __SYCL_FALLTHROUGH;
  case 1:
    getImageInfo(Context, PI_IMAGE_INFO_WIDTH, MRange[0], Mem);
  }
}

template <int Dimensions>
void *image_impl<Dimensions>::allocateMem(ContextImplPtr Context,
                                          bool InitFromUserData, void *HostPtr,
                                          RT::PiEvent &OutEventToWait) {
  bool HostPtrReadOnly = false;
  BaseT::determineHostPtr(Context, InitFromUserData, HostPtr, HostPtrReadOnly);

  RT::PiMemImageDesc Desc = getImageDesc(HostPtr != nullptr);
  assert(checkImageDesc(Desc, Context, HostPtr) &&
         "The check an image desc failed.");

  RT::PiMemImageFormat Format = getImageFormat();
  assert(checkImageFormat(Format, Context) &&
         "The check an image format failed.");

  return MemoryManager::allocateMemImage(
      std::move(Context), this, HostPtr, HostPtrReadOnly, BaseT::getSize(),
      Desc, Format, BaseT::MInteropEvent, BaseT::MInteropContext, MProps,
      OutEventToWait);
}

template <int Dimensions>
bool image_impl<Dimensions>::checkImageDesc(const RT::PiMemImageDesc &Desc,
                                            ContextImplPtr Context,
                                            void *UserPtr) {
  if (checkAny(Desc.image_type, PI_MEM_TYPE_IMAGE1D, PI_MEM_TYPE_IMAGE1D_ARRAY,
               PI_MEM_TYPE_IMAGE2D_ARRAY, PI_MEM_TYPE_IMAGE2D) &&
      !checkImageValueRange<info::device::image2d_max_width>(
          getDevices(Context), Desc.image_width))
    throw invalid_parameter_error(
        "For a 1D/2D image/image array, the width must be a Value >= 1 and "
        "<= CL_DEVICE_IMAGE2D_MAX_WIDTH.",
        PI_INVALID_VALUE);

  if (checkAny(Desc.image_type, PI_MEM_TYPE_IMAGE3D) &&
      !checkImageValueRange<info::device::image3d_max_width>(
          getDevices(Context), Desc.image_width))
    throw invalid_parameter_error(
        "For a 3D image, the width must be a Value >= 1 and <= "
        "CL_DEVICE_IMAGE3D_MAX_WIDTH",
        PI_INVALID_VALUE);

  if (checkAny(Desc.image_type, PI_MEM_TYPE_IMAGE2D,
               PI_MEM_TYPE_IMAGE2D_ARRAY) &&
      !checkImageValueRange<info::device::image2d_max_height>(
          getDevices(Context), Desc.image_height))
    throw invalid_parameter_error("For a 2D image or image array, the height "
                                  "must be a Value >= 1 and <= "
                                  "CL_DEVICE_IMAGE2D_MAX_HEIGHT",
                                  PI_INVALID_VALUE);

  if (checkAny(Desc.image_type, PI_MEM_TYPE_IMAGE3D) &&
      !checkImageValueRange<info::device::image3d_max_height>(
          getDevices(Context), Desc.image_height))
    throw invalid_parameter_error(
        "For a 3D image, the heightmust be a Value >= 1 and <= "
        "CL_DEVICE_IMAGE3D_MAX_HEIGHT",
        PI_INVALID_VALUE);

  if (checkAny(Desc.image_type, PI_MEM_TYPE_IMAGE3D) &&
      !checkImageValueRange<info::device::image3d_max_depth>(
          getDevices(Context), Desc.image_depth))
    throw invalid_parameter_error(
        "For a 3D image, the depth must be a Value >= 1 and <= "
        "CL_DEVICE_IMAGE3D_MAX_DEPTH",
        PI_INVALID_VALUE);

  if (checkAny(Desc.image_type, PI_MEM_TYPE_IMAGE1D_ARRAY,
               PI_MEM_TYPE_IMAGE2D_ARRAY) &&
      !checkImageValueRange<info::device::image_max_array_size>(
          getDevices(Context), Desc.image_array_size))
    throw invalid_parameter_error(
        "For a 1D and 2D image array, the array_size must be a "
        "Value >= 1 and <= CL_DEVICE_IMAGE_MAX_ARRAY_SIZE.",
        PI_INVALID_VALUE);

  if ((nullptr == UserPtr) && (0 != Desc.image_row_pitch))
    throw invalid_parameter_error(
        "The row_pitch must be 0 if host_ptr is nullptr.", PI_INVALID_VALUE);

  if ((nullptr == UserPtr) && (0 != Desc.image_slice_pitch))
    throw invalid_parameter_error(
        "The slice_pitch must be 0 if host_ptr is nullptr.", PI_INVALID_VALUE);

  if (0 != Desc.num_mip_levels)
    throw invalid_parameter_error("The mip_levels must be 0.",
                                  PI_INVALID_VALUE);

  if (0 != Desc.num_samples)
    throw invalid_parameter_error("The num_samples must be 0.",
                                  PI_INVALID_VALUE);

  if (nullptr != Desc.buffer)
    throw invalid_parameter_error(
        "The buffer must be nullptr, because SYCL does not support "
        "image creation from memory objects.",
        PI_INVALID_VALUE);

  return true;
}

template <int Dimensions>
bool image_impl<Dimensions>::checkImageFormat(
    const RT::PiMemImageFormat &Format, ContextImplPtr Context) {
  (void)Context;
  if (checkAny(Format.image_channel_order, PI_IMAGE_CHANNEL_ORDER_INTENSITY,
               PI_IMAGE_CHANNEL_ORDER_LUMINANCE) &&
      !checkAny(
          Format.image_channel_data_type, PI_IMAGE_CHANNEL_TYPE_UNORM_INT8,
          PI_IMAGE_CHANNEL_TYPE_UNORM_INT16, PI_IMAGE_CHANNEL_TYPE_SNORM_INT8,
          PI_IMAGE_CHANNEL_TYPE_SNORM_INT16, PI_IMAGE_CHANNEL_TYPE_HALF_FLOAT,
          PI_IMAGE_CHANNEL_TYPE_FLOAT))
    throw invalid_parameter_error(
        "CL_INTENSITY or CL_LUMINANCE format can only be used if channel "
        "data type = CL_UNORM_INT8, CL_UNORM_INT16, CL_SNORM_INT8, "
        "CL_SNORM_INT16, CL_HALF_FLOAT, or CL_FLOAT.",
        PI_INVALID_VALUE);

  if (checkAny(Format.image_channel_data_type,
               PI_IMAGE_CHANNEL_TYPE_UNORM_SHORT_565,
               PI_IMAGE_CHANNEL_TYPE_UNORM_SHORT_555,
               PI_IMAGE_CHANNEL_TYPE_UNORM_INT_101010) &&
      !checkAny(Format.image_channel_order, PI_IMAGE_CHANNEL_ORDER_RGB,
                PI_IMAGE_CHANNEL_ORDER_RGBx))
    throw invalid_parameter_error(
        "type = CL_UNORM_SHORT_565, CL_UNORM_SHORT_555 or "
        "CL_UNORM_INT_101010."
        "These channel types can only be used with CL_RGB or CL_RGBx channel "
        "order.",
        PI_INVALID_VALUE);

  if (checkAny(Format.image_channel_order, PI_IMAGE_CHANNEL_ORDER_ARGB,
               PI_IMAGE_CHANNEL_ORDER_BGRA, PI_IMAGE_CHANNEL_ORDER_ABGR) &&
      !checkAny(
          Format.image_channel_data_type, PI_IMAGE_CHANNEL_TYPE_UNORM_INT8,
          PI_IMAGE_CHANNEL_TYPE_SNORM_INT8, PI_IMAGE_CHANNEL_TYPE_SIGNED_INT8,
          PI_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8))
    throw invalid_parameter_error(
        "CL_ARGB, CL_BGRA, CL_ABGR	These formats can only be used if "
        "channel data type = CL_UNORM_INT8, CL_SNORM_INT8, CL_SIGNED_INT8 "
        "or CL_UNSIGNED_INT8.",
        PI_INVALID_VALUE);

  return true;
}

template <int Dimensions>
std::vector<device>
image_impl<Dimensions>::getDevices(const ContextImplPtr Context) {
  return Context->get_info<info::context::devices>();
}

template class image_impl<1>;
template class image_impl<2>;
template class image_impl<3>;

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
