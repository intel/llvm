//==------------ image_impl.cpp --------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/context_impl.hpp>
#include <detail/image_impl.hpp>
#include <detail/memory_manager.hpp>
#include <detail/xpti_registry.hpp>
#include <sycl/detail/ur.hpp>

#include <algorithm>
#include <vector>

namespace sycl {
inline namespace _V1 {
namespace detail {
#ifdef XPTI_ENABLE_INSTRUMENTATION
uint8_t GImageStreamID;
#endif

template <typename Param>
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

ur_image_channel_order_t convertChannelOrder(image_channel_order Order) {
  switch (Order) {
  case image_channel_order::a:
    return UR_IMAGE_CHANNEL_ORDER_A;
  case image_channel_order::r:
    return UR_IMAGE_CHANNEL_ORDER_R;
  case image_channel_order::rx:
    return UR_IMAGE_CHANNEL_ORDER_RX;
  case image_channel_order::rg:
    return UR_IMAGE_CHANNEL_ORDER_RG;
  case image_channel_order::rgx:
    return UR_IMAGE_CHANNEL_ORDER_RGX;
  case image_channel_order::ra:
    return UR_IMAGE_CHANNEL_ORDER_RA;
  case image_channel_order::rgb:
    return UR_IMAGE_CHANNEL_ORDER_RGB;
  case image_channel_order::rgbx:
    return UR_IMAGE_CHANNEL_ORDER_RGBX;
  case image_channel_order::rgba:
    return UR_IMAGE_CHANNEL_ORDER_RGBA;
  case image_channel_order::argb:
    return UR_IMAGE_CHANNEL_ORDER_ARGB;
  case image_channel_order::bgra:
    return UR_IMAGE_CHANNEL_ORDER_BGRA;
  case image_channel_order::intensity:
    return UR_IMAGE_CHANNEL_ORDER_INTENSITY;
  case image_channel_order::luminance:
    return UR_IMAGE_CHANNEL_ORDER_LUMINANCE;
  case image_channel_order::abgr:
    return UR_IMAGE_CHANNEL_ORDER_ABGR;
  case image_channel_order::ext_oneapi_srgba:
    return UR_IMAGE_CHANNEL_ORDER_SRGBA;
  }
  assert(false && "Unhandled image_channel_order");
  return static_cast<ur_image_channel_order_t>(0);
}

image_channel_order convertChannelOrder(ur_image_channel_order_t Order) {
  switch (Order) {
  case UR_IMAGE_CHANNEL_ORDER_A:
    return image_channel_order::a;
  case UR_IMAGE_CHANNEL_ORDER_R:
    return image_channel_order::r;
  case UR_IMAGE_CHANNEL_ORDER_RX:
    return image_channel_order::rx;
  case UR_IMAGE_CHANNEL_ORDER_RG:
    return image_channel_order::rg;
  case UR_IMAGE_CHANNEL_ORDER_RGX:
    return image_channel_order::rgx;
  case UR_IMAGE_CHANNEL_ORDER_RA:
    return image_channel_order::ra;
  case UR_IMAGE_CHANNEL_ORDER_RGB:
    return image_channel_order::rgb;
  case UR_IMAGE_CHANNEL_ORDER_RGBX:
    return image_channel_order::rgbx;
  case UR_IMAGE_CHANNEL_ORDER_RGBA:
    return image_channel_order::rgba;
  case UR_IMAGE_CHANNEL_ORDER_ARGB:
    return image_channel_order::argb;
  case UR_IMAGE_CHANNEL_ORDER_BGRA:
    return image_channel_order::bgra;
  case UR_IMAGE_CHANNEL_ORDER_INTENSITY:
    return image_channel_order::intensity;
  case UR_IMAGE_CHANNEL_ORDER_LUMINANCE:
    return image_channel_order::luminance;
  case UR_IMAGE_CHANNEL_ORDER_ABGR:
    return image_channel_order::abgr;
  case UR_IMAGE_CHANNEL_ORDER_SRGBA:
    return image_channel_order::ext_oneapi_srgba;
  default:
    assert(false && "Unhandled image_channel_order");
  }
  return static_cast<image_channel_order>(0);
}

ur_image_channel_type_t convertChannelType(image_channel_type Type) {
  switch (Type) {
  case image_channel_type::snorm_int8:
    return UR_IMAGE_CHANNEL_TYPE_SNORM_INT8;
  case image_channel_type::snorm_int16:
    return UR_IMAGE_CHANNEL_TYPE_SNORM_INT16;
  case image_channel_type::unorm_int8:
    return UR_IMAGE_CHANNEL_TYPE_UNORM_INT8;
  case image_channel_type::unorm_int16:
    return UR_IMAGE_CHANNEL_TYPE_UNORM_INT16;
  case image_channel_type::unorm_short_565:
    return UR_IMAGE_CHANNEL_TYPE_UNORM_SHORT_565;
  case image_channel_type::unorm_short_555:
    return UR_IMAGE_CHANNEL_TYPE_UNORM_SHORT_555;
  case image_channel_type::unorm_int_101010:
    return UR_IMAGE_CHANNEL_TYPE_INT_101010;
  case image_channel_type::signed_int8:
    return UR_IMAGE_CHANNEL_TYPE_SIGNED_INT8;
  case image_channel_type::signed_int16:
    return UR_IMAGE_CHANNEL_TYPE_SIGNED_INT16;
  case image_channel_type::signed_int32:
    return UR_IMAGE_CHANNEL_TYPE_SIGNED_INT32;
  case image_channel_type::unsigned_int8:
    return UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8;
  case image_channel_type::unsigned_int16:
    return UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16;
  case image_channel_type::unsigned_int32:
    return UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32;
  case image_channel_type::fp16:
    return UR_IMAGE_CHANNEL_TYPE_HALF_FLOAT;
  case image_channel_type::fp32:
    return UR_IMAGE_CHANNEL_TYPE_FLOAT;
  }
  assert(false && "Unhandled image_channel_order");
  return static_cast<ur_image_channel_type_t>(0);
}

image_channel_type convertChannelType(ur_image_channel_type_t Type) {
  switch (Type) {
  case UR_IMAGE_CHANNEL_TYPE_SNORM_INT8:
    return image_channel_type::snorm_int8;
  case UR_IMAGE_CHANNEL_TYPE_SNORM_INT16:
    return image_channel_type::snorm_int16;
  case UR_IMAGE_CHANNEL_TYPE_UNORM_INT8:
    return image_channel_type::unorm_int8;
  case UR_IMAGE_CHANNEL_TYPE_UNORM_INT16:
    return image_channel_type::unorm_int16;
  case UR_IMAGE_CHANNEL_TYPE_UNORM_SHORT_565:
    return image_channel_type::unorm_short_565;
  case UR_IMAGE_CHANNEL_TYPE_UNORM_SHORT_555:
    return image_channel_type::unorm_short_555;
  case UR_IMAGE_CHANNEL_TYPE_INT_101010:
    return image_channel_type::unorm_int_101010;
  case UR_IMAGE_CHANNEL_TYPE_SIGNED_INT8:
    return image_channel_type::signed_int8;
  case UR_IMAGE_CHANNEL_TYPE_SIGNED_INT16:
    return image_channel_type::signed_int16;
  case UR_IMAGE_CHANNEL_TYPE_SIGNED_INT32:
    return image_channel_type::signed_int32;
  case UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8:
    return image_channel_type::unsigned_int8;
  case UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16:
    return image_channel_type::unsigned_int16;
  case UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32:
    return image_channel_type::unsigned_int32;
  case UR_IMAGE_CHANNEL_TYPE_HALF_FLOAT:
    return image_channel_type::fp16;
  case UR_IMAGE_CHANNEL_TYPE_FLOAT:
    return image_channel_type::fp32;
  default:
    assert(false && "Unhandled image_channel_order");
  }
  return static_cast<image_channel_type>(0);
}

template <typename T>
static void getImageInfo(const ContextImplPtr Context, ur_image_info_t Info,
                         T &Dest, ur_mem_handle_t InteropMemObject) {
  const PluginPtr &Plugin = Context->getPlugin();
  Plugin->call(urMemImageGetInfo, InteropMemObject, Info, sizeof(T), &Dest,
               nullptr);
}

image_impl::image_impl(cl_mem MemObject, const context &SyclContext,
                       event AvailableEvent,
                       std::unique_ptr<SYCLMemObjAllocator> Allocator,
                       uint8_t Dimensions)
    : BaseT(MemObject, SyclContext, std::move(AvailableEvent),
            std::move(Allocator)),
      MDimensions(Dimensions), MRange({0, 0, 0}) {
  ur_mem_handle_t Mem = ur::cast<ur_mem_handle_t>(BaseT::MInteropMemObject);
  const ContextImplPtr Context = getSyclObjImpl(SyclContext);
  const PluginPtr &Plugin = Context->getPlugin();
  Plugin->call(urMemGetInfo, Mem, UR_MEM_INFO_SIZE, sizeof(size_t),
               &(BaseT::MSizeInBytes), nullptr);

  ur_image_format_t Format;
  getImageInfo(Context, UR_IMAGE_INFO_FORMAT, Format, Mem);
  MOrder = detail::convertChannelOrder(Format.channelOrder);
  MType = detail::convertChannelType(Format.channelType);
  MNumChannels = getImageNumberChannels(MOrder);

  getImageInfo(Context, UR_IMAGE_INFO_ELEMENT_SIZE, MElementSize, Mem);
  assert(getImageElementSize(MNumChannels, MType) == MElementSize);

  getImageInfo(Context, UR_IMAGE_INFO_ROW_PITCH, MRowPitch, Mem);
  getImageInfo(Context, UR_IMAGE_INFO_SLICE_PITCH, MSlicePitch, Mem);

  switch (MDimensions) {
  case 3:
    getImageInfo(Context, UR_IMAGE_INFO_DEPTH, MRange[2], Mem);
    [[fallthrough]];
  case 2:
    getImageInfo(Context, UR_IMAGE_INFO_HEIGHT, MRange[1], Mem);
    [[fallthrough]];
  case 1:
    getImageInfo(Context, UR_IMAGE_INFO_WIDTH, MRange[0], Mem);
  }
}

image_impl::image_impl(ur_native_handle_t MemObject, const context &SyclContext,
                       event AvailableEvent,
                       std::unique_ptr<SYCLMemObjAllocator> Allocator,
                       uint8_t Dimensions, image_channel_order Order,
                       image_channel_type Type, bool OwnNativeHandle,
                       range<3> Range3WithOnes)
    : BaseT(MemObject, SyclContext, OwnNativeHandle, std::move(AvailableEvent),
            std::move(Allocator),
            ur_image_format_t{detail::convertChannelOrder(Order),
                              detail::convertChannelType(Type)},
            Range3WithOnes, Dimensions,
            getImageElementSize(getImageNumberChannels(Order), Type)),
      MDimensions(Dimensions), MRange(Range3WithOnes) {
  MOrder = Order;
  MType = Type;
  MNumChannels = getImageNumberChannels(MOrder);
  MElementSize = getImageElementSize(MNumChannels, Type);
  setPitches(); // sets MRowPitch, MSlice and BaseT::MSizeInBytes
}

void *image_impl::allocateMem(ContextImplPtr Context, bool InitFromUserData,
                              void *HostPtr,
                              ur_event_handle_t &OutEventToWait) {
  bool HostPtrReadOnly = false;
  BaseT::determineHostPtr(Context, InitFromUserData, HostPtr, HostPtrReadOnly);

  ur_image_desc_t Desc = getImageDesc(HostPtr != nullptr);
  assert(checkImageDesc(Desc, Context, HostPtr) &&
         "The check an image desc failed.");

  ur_image_format_t Format = getImageFormat();
  assert(checkImageFormat(Format, Context) &&
         "The check an image format failed.");

  return MemoryManager::allocateMemImage(
      std::move(Context), this, HostPtr, HostPtrReadOnly,
      BaseT::getSizeInBytes(), Desc, Format, BaseT::MInteropEvent,
      BaseT::MInteropContext, MProps, OutEventToWait);
}

bool image_impl::checkImageDesc(const ur_image_desc_t &Desc,
                                ContextImplPtr Context, void *UserPtr) {
  if (checkAny(Desc.type, UR_MEM_TYPE_IMAGE1D, UR_MEM_TYPE_IMAGE1D_ARRAY,
               UR_MEM_TYPE_IMAGE2D_ARRAY, UR_MEM_TYPE_IMAGE2D) &&
      !checkImageValueRange<info::device::image2d_max_width>(
          getDevices(Context), Desc.width))
    throw exception(make_error_code(errc::invalid),
                    "For a 1D/2D image/image array, the width must be a Value "
                    ">= 1 and <= info::device::image2d_max_width");

  if (checkAny(Desc.type, UR_MEM_TYPE_IMAGE3D) &&
      !checkImageValueRange<info::device::image3d_max_width>(
          getDevices(Context), Desc.width))
    throw exception(make_error_code(errc::invalid),
                    "For a 3D image, the width must be a Value >= 1 and <= "
                    "info::device::image3d_max_width");

  if (checkAny(Desc.type, UR_MEM_TYPE_IMAGE2D, UR_MEM_TYPE_IMAGE2D_ARRAY) &&
      !checkImageValueRange<info::device::image2d_max_height>(
          getDevices(Context), Desc.height))
    throw exception(make_error_code(errc::invalid),
                    "For a 2D image or image array, the height must be a Value "
                    ">= 1 and <= info::device::image2d_max_height");

  if (checkAny(Desc.type, UR_MEM_TYPE_IMAGE3D) &&
      !checkImageValueRange<info::device::image3d_max_height>(
          getDevices(Context), Desc.height))
    throw exception(make_error_code(errc::invalid),
                    "For a 3D image, the heightmust be a Value >= 1 and <= "
                    "info::device::image3d_max_height");

  if (checkAny(Desc.type, UR_MEM_TYPE_IMAGE3D) &&
      !checkImageValueRange<info::device::image3d_max_depth>(
          getDevices(Context), Desc.depth))
    throw exception(make_error_code(errc::invalid),
                    "For a 3D image, the depth must be a Value >= 1 and <= "
                    "info::device::image2d_max_depth");

  if (checkAny(Desc.type, UR_MEM_TYPE_IMAGE1D_ARRAY,
               UR_MEM_TYPE_IMAGE2D_ARRAY) &&
      !checkImageValueRange<info::device::image_max_array_size>(
          getDevices(Context), Desc.arraySize))
    throw exception(make_error_code(errc::invalid),
                    "For a 1D and 2D image array, the array_size must be a "
                    "Value >= 1 and <= info::device::image_max_array_size.");

  if ((nullptr == UserPtr) && (0 != Desc.rowPitch))
    throw exception(make_error_code(errc::invalid),
                    "The row_pitch must be 0 if host_ptr is nullptr.");

  if ((nullptr == UserPtr) && (0 != Desc.slicePitch))
    throw exception(make_error_code(errc::invalid),
                    "The slice_pitch must be 0 if host_ptr is nullptr.");

  if (0 != Desc.numMipLevel)
    throw exception(make_error_code(errc::invalid),
                    "The mip_levels must be 0.");

  if (0 != Desc.numSamples)
    throw exception(make_error_code(errc::invalid),
                    "The num_samples must be 0.");

  return true;
}

bool image_impl::checkImageFormat(const ur_image_format_t &Format,
                                  ContextImplPtr Context) {
  (void)Context;
  if (checkAny(Format.channelOrder, UR_IMAGE_CHANNEL_ORDER_INTENSITY,
               UR_IMAGE_CHANNEL_ORDER_LUMINANCE) &&
      !checkAny(Format.channelType, UR_IMAGE_CHANNEL_TYPE_UNORM_INT8,
                UR_IMAGE_CHANNEL_TYPE_UNORM_INT16,
                UR_IMAGE_CHANNEL_TYPE_SNORM_INT8,
                UR_IMAGE_CHANNEL_TYPE_SNORM_INT16,
                UR_IMAGE_CHANNEL_TYPE_HALF_FLOAT, UR_IMAGE_CHANNEL_TYPE_FLOAT))
    throw exception(
        make_error_code(errc::invalid),
        "CL_INTENSITY or CL_LUMINANCE format can only be used if channel "
        "data type = CL_UNORM_INT8, CL_UNORM_INT16, CL_SNORM_INT8, "
        "CL_SNORM_INT16, CL_HALF_FLOAT, or CL_FLOAT.");

  if (checkAny(Format.channelType, UR_IMAGE_CHANNEL_TYPE_UNORM_SHORT_565,
               UR_IMAGE_CHANNEL_TYPE_UNORM_SHORT_555,
               UR_IMAGE_CHANNEL_TYPE_INT_101010) &&
      !checkAny(Format.channelOrder, UR_IMAGE_CHANNEL_ORDER_RGB,
                UR_IMAGE_CHANNEL_ORDER_RGBX))
    throw exception(
        make_error_code(errc::invalid),
        "type = CL_UNORM_SHORT_565, CL_UNORM_SHORT_555 or "
        "CL_UNORM_INT_101010."
        "These channel types can only be used with CL_RGB or CL_RGBx channel "
        "order.");

  if (checkAny(Format.channelOrder, UR_IMAGE_CHANNEL_ORDER_ARGB,
               UR_IMAGE_CHANNEL_ORDER_BGRA, UR_IMAGE_CHANNEL_ORDER_ABGR) &&
      !checkAny(Format.channelType, UR_IMAGE_CHANNEL_TYPE_UNORM_INT8,
                UR_IMAGE_CHANNEL_TYPE_SNORM_INT8,
                UR_IMAGE_CHANNEL_TYPE_SIGNED_INT8,
                UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8))
    throw exception(
        make_error_code(errc::invalid),
        "CL_ARGB, CL_BGRA, CL_ABGR	These formats can only be used if "
        "channel data type = CL_UNORM_INT8, CL_SNORM_INT8, CL_SIGNED_INT8 "
        "or CL_UNSIGNED_INT8.");

  return true;
}

std::vector<device> image_impl::getDevices(const ContextImplPtr Context) {
  if (!Context)
    return {};
  return Context->get_info<info::context::devices>();
}

void image_impl::sampledImageConstructorNotification(
    const detail::code_location &CodeLoc, void *UserObj, const void *HostObj,
    uint32_t Dim, size_t Range[3], image_format Format,
    const image_sampler &Sampler) {
  XPTIRegistry::sampledImageConstructorNotification(
      UserObj, CodeLoc, HostObj, Dim, Range, (uint32_t)Format,
      (uint32_t)Sampler.addressing, (uint32_t)Sampler.coordinate,
      (uint32_t)Sampler.filtering);
}

void image_impl::sampledImageDestructorNotification(void *UserObj) {
  XPTIRegistry::sampledImageDestructorNotification(UserObj);
}

void image_impl::unsampledImageConstructorNotification(
    const detail::code_location &CodeLoc, void *UserObj, const void *HostObj,
    uint32_t Dim, size_t Range[3], image_format Format) {
  XPTIRegistry::unsampledImageConstructorNotification(
      UserObj, CodeLoc, HostObj, Dim, Range, (uint32_t)Format);
}

void image_impl::unsampledImageDestructorNotification(void *UserObj) {
  XPTIRegistry::unsampledImageDestructorNotification(UserObj);
}

} // namespace detail
} // namespace _V1
} // namespace sycl
