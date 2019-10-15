//==------------ image_impl.hpp --------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/aligned_allocator.hpp>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/generic_type_traits.hpp>
#include <CL/sycl/detail/memory_manager.hpp>
#include <CL/sycl/detail/scheduler/scheduler.hpp>
#include <CL/sycl/detail/sycl_mem_obj_t.hpp>
#include <CL/sycl/event.hpp>
#include <CL/sycl/property_list.hpp>
#include <CL/sycl/range.hpp>
#include <CL/sycl/stl.hpp>

namespace cl {
namespace sycl {

// forward declarations
enum class image_channel_order : unsigned int;
enum class image_channel_type : unsigned int;

template <int Dimensions, typename AllocatorT> class image;
template <typename DataT, int Dimensions, access::mode AccessMode,
          access::target AccessTarget, access::placeholder IsPlaceholder>
class accessor;
class handler;

namespace detail {

// utility functions and typedefs for image_impl
using image_allocator = aligned_allocator<byte>;

// utility function: Returns the Number of Channels for a given Order.
uint8_t getImageNumberChannels(image_channel_order Order);

// utility function: Returns the number of bytes per image element
uint8_t getImageElementSize(uint8_t NumChannels, image_channel_type Type);

RT::PiMemImageChannelOrder convertChannelOrder(image_channel_order Order);

image_channel_order convertChannelOrder(RT::PiMemImageChannelOrder Order);

RT::PiMemImageChannelType convertChannelType(image_channel_type Type);

image_channel_type convertChannelType(RT::PiMemImageChannelType Type);

// validImageDataT: cl_int4, cl_uint4, cl_float4, cl_half4
template <typename T>
using is_validImageDataT = typename detail::is_contained<
    T, type_list<cl_int4, cl_uint4, cl_float4, cl_half4>>::type;

template <typename DataT>
using EnableIfImgAccDataT =
    typename std::enable_if<is_validImageDataT<DataT>::value, DataT>::type;

template <int Dimensions, typename AllocatorT = image_allocator>
class image_impl final : public SYCLMemObjT<AllocatorT> {
  using BaseT = SYCLMemObjT<AllocatorT>;
  using typename BaseT::MemObjType;

private:
  template <bool B>
  using EnableIfPitchT =
      typename std::enable_if<B, range<Dimensions - 1>>::type;
  static_assert(Dimensions >= 1 || Dimensions <= 3,
                "Dimensions of cl::sycl::image can be 1, 2 or 3");

  void setPitches() {
    size_t WHD[3] = {1, 1, 1}; // Width, Height, Depth.
    for (int I = 0; I < Dimensions; I++)
      WHD[I] = MRange[I];
    MRowPitch = MElementSize * WHD[0];
    MSlicePitch = MRowPitch * WHD[1];
    BaseT::MSizeInBytes = MSlicePitch * WHD[2];
  }

  template <bool B = (Dimensions > 1)>
  void setPitches(const EnableIfPitchT<B> Pitch) {
    MRowPitch = Pitch[0];
    MSlicePitch =
        (Dimensions == 3) ? Pitch[1] : MRowPitch; // Dimensions will be 2/3.
    // NumSlices is depth when dim==3, and height when dim==2.
    size_t NumSlices =
        (Dimensions == 3) ? MRange[2] : MRange[1]; // Dimensions will be 2/3.
    BaseT::MSizeInBytes = MSlicePitch * NumSlices;
  }

public:
  image_impl(image_channel_order Order, image_channel_type Type,
             const range<Dimensions> &ImageRange,
             const property_list &PropList = {})
      : image_impl((void *)nullptr, Order, Type, ImageRange, PropList) {}

  image_impl(image_channel_order Order, image_channel_type Type,
             const range<Dimensions> &ImageRange, AllocatorT Allocator,
             const property_list &PropList = {})
      : image_impl((void *)nullptr, Order, Type, ImageRange, Allocator,
                   PropList) {}

  template <bool B = (Dimensions > 1)>
  image_impl(image_channel_order Order, image_channel_type Type,
             const range<Dimensions> &ImageRange,
             const EnableIfPitchT<B> &Pitch, const property_list &PropList = {})
      : image_impl((void *)nullptr, Order, Type, ImageRange, Pitch, PropList) {}

  template <bool B = (Dimensions > 1)>
  image_impl(image_channel_order Order, image_channel_type Type,
             const range<Dimensions> &ImageRange,
             const EnableIfPitchT<B> &Pitch, AllocatorT Allocator,
             const property_list &PropList = {})
      : image_impl((void *)nullptr, Order, Type, ImageRange, Pitch, Allocator,
                   PropList) {}

  image_impl(void *HData, image_channel_order Order, image_channel_type Type,
             const range<Dimensions> &ImageRange,
             const property_list &PropList = {})
      : BaseT(PropList), MRange(ImageRange), MOrder(Order), MType(Type),
        MNumChannels(getImageNumberChannels(MOrder)),
        MElementSize(getImageElementSize(MNumChannels, MType)) {
    setPitches();
    BaseT::handleHostData(HData, detail::getNextPowerOfTwo(MElementSize));
  }

  image_impl(void *HData, image_channel_order Order, image_channel_type Type,
             const range<Dimensions> &ImageRange, AllocatorT Allocator,
             const property_list &PropList = {})
      : BaseT(PropList, Allocator), MRange(ImageRange), MOrder(Order),
        MType(Type), MNumChannels(getImageNumberChannels(MOrder)),
        MElementSize(getImageElementSize(MNumChannels, MType)) {
    setPitches();
    BaseT::handleHostData(HData, detail::getNextPowerOfTwo(MElementSize));
  }

  image_impl(const void *HData, image_channel_order Order,
             image_channel_type Type, const range<Dimensions> &ImageRange,
             const property_list &PropList = {})
      : BaseT(PropList), MRange(ImageRange), MOrder(Order), MType(Type),
        MNumChannels(getImageNumberChannels(MOrder)),
        MElementSize(getImageElementSize(MNumChannels, MType)) {
    setPitches();
    BaseT::handleHostData(HData, detail::getNextPowerOfTwo(MElementSize));
  }

  image_impl(const void *HData, image_channel_order Order,
             image_channel_type Type, const range<Dimensions> &ImageRange,
             AllocatorT Allocator, const property_list &PropList = {})
      : BaseT(PropList, Allocator), MRange(ImageRange), MOrder(Order),
        MType(Type), MNumChannels(getImageNumberChannels(MOrder)),
        MElementSize(getImageElementSize(MNumChannels, MType)) {
    setPitches();
    BaseT::handleHostData(HData, detail::getNextPowerOfTwo(MElementSize));
  }

  template <bool B = (Dimensions > 1)>
  image_impl(void *HData, image_channel_order Order, image_channel_type Type,
             const range<Dimensions> &ImageRange,
             const EnableIfPitchT<B> &Pitch, const property_list &PropList = {})
      : BaseT(PropList), MRange(ImageRange), MOrder(Order), MType(Type),
        MNumChannels(getImageNumberChannels(MOrder)),
        MElementSize(getImageElementSize(MNumChannels, MType)) {
    setPitches(Pitch);
    BaseT::handleHostData(HData, detail::getNextPowerOfTwo(MElementSize));
  }

  template <bool B = (Dimensions > 1)>
  image_impl(void *HData, image_channel_order Order, image_channel_type Type,
             const range<Dimensions> &ImageRange,
             const EnableIfPitchT<B> &Pitch, AllocatorT Allocator,
             const property_list &PropList = {})
      : BaseT(PropList, Allocator), MRange(ImageRange), MOrder(Order),
        MType(Type), MNumChannels(getImageNumberChannels(MOrder)),
        MElementSize(getImageElementSize(MNumChannels, MType)) {
    setPitches(Pitch);
    BaseT::handleHostData(HData, detail::getNextPowerOfTwo(MElementSize));
  }

  image_impl(shared_ptr_class<void> &HData, image_channel_order Order,
             image_channel_type Type, const range<Dimensions> &ImageRange,
             const property_list &PropList = {})
      : BaseT(PropList), MRange(ImageRange), MOrder(Order), MType(Type),
        MNumChannels(getImageNumberChannels(MOrder)),
        MElementSize(getImageElementSize(MNumChannels, MType)) {
    setPitches();
    BaseT::handleHostData(HData, detail::getNextPowerOfTwo(MElementSize));
  }

  image_impl(shared_ptr_class<void> &HData, image_channel_order Order,
             image_channel_type Type, const range<Dimensions> &ImageRange,
             AllocatorT Allocator, const property_list &PropList = {})
      : BaseT(PropList, Allocator), MRange(ImageRange), MOrder(Order),
        MType(Type), MNumChannels(getImageNumberChannels(MOrder)),
        MElementSize(getImageElementSize(MNumChannels, MType)) {
    setPitches();
    BaseT::handleHostData(HData, detail::getNextPowerOfTwo(MElementSize));
  }

  /* Available only when: Dimensions > 1 */
  template <bool B = (Dimensions > 1)>
  image_impl(shared_ptr_class<void> &HData, image_channel_order Order,
             image_channel_type Type, const range<Dimensions> &ImageRange,
             const EnableIfPitchT<B> &Pitch, const property_list &PropList = {})
      : BaseT(PropList), MRange(ImageRange), MOrder(Order), MType(Type),
        MNumChannels(getImageNumberChannels(MOrder)),
        MElementSize(getImageElementSize(MNumChannels, MType)) {
    setPitches(Pitch);
    BaseT::handleHostData(HData, detail::getNextPowerOfTwo(MElementSize));
  }

  /* Available only when: Dimensions > 1 */
  template <bool B = (Dimensions > 1)>
  image_impl(shared_ptr_class<void> &HData, image_channel_order Order,
             image_channel_type Type, const range<Dimensions> &ImageRange,
             const EnableIfPitchT<B> &Pitch, AllocatorT Allocator,
             const property_list &PropList = {})
      : BaseT(PropList, Allocator), MRange(ImageRange), MOrder(Order),
        MType(Type), MNumChannels(getImageNumberChannels(MOrder)),
        MElementSize(getImageElementSize(MNumChannels, MType)) {
    setPitches(Pitch);
    BaseT::handleHostData(HData, detail::getNextPowerOfTwo(MElementSize));
  }

  image_impl(cl_mem MemObject, const context &SyclContext,
             event AvailableEvent = {})
      : BaseT(MemObject, SyclContext, std::move(AvailableEvent)),
        MRange(InitializedVal<Dimensions, range>::template get<0>()) {
    RT::PiMem Mem = pi::cast<RT::PiMem>(BaseT::MInteropMemObject);
    PI_CALL(RT::piMemGetInfo(Mem, CL_MEM_SIZE, sizeof(size_t),
                             &(BaseT::MSizeInBytes), nullptr));

    RT::PiMemImageFormat Format;
    getImageInfo(PI_IMAGE_INFO_FORMAT, Format);
    MOrder = detail::convertChannelOrder(Format.image_channel_order);
    MType = detail::convertChannelType(Format.image_channel_data_type);
    MNumChannels = getImageNumberChannels(MOrder);

    getImageInfo(PI_IMAGE_INFO_ELEMENT_SIZE, MElementSize);
    assert(getImageElementSize(MNumChannels, MType) == MElementSize);

    getImageInfo(PI_IMAGE_INFO_ROW_PITCH, MRowPitch);
    getImageInfo(PI_IMAGE_INFO_SLICE_PITCH, MSlicePitch);

    switch (Dimensions) {
    case 3:
      getImageInfo(PI_IMAGE_INFO_DEPTH, MRange[2]);
    case 2:
      getImageInfo(PI_IMAGE_INFO_HEIGHT, MRange[1]);
    case 1:
      getImageInfo(PI_IMAGE_INFO_WIDTH, MRange[0]);
    }
  }

  // Return a range object representing the size of the image in terms of the
  // number of elements in each dimension as passed to the constructor
  range<Dimensions> get_range() const { return MRange; }

  // Return a range object representing the pitch of the image in bytes.
  // Available only when: Dimensions == 2.
  template <bool B = (Dimensions == 2)>
  typename std::enable_if<B, range<1>>::type get_pitch() const {
    range<1> Temp = range<1>(MRowPitch);
    return Temp;
  }

  // Return a range object representing the pitch of the image in bytes.
  // Available only when: Dimensions == 3.
  template <bool B = (Dimensions == 3)>
  typename std::enable_if<B, range<2>>::type get_pitch() const {
    range<2> Temp = range<2>(MRowPitch, MSlicePitch);
    return Temp;
  }

  // Returns the total number of elements in the image
  size_t get_count() const { return MRange.size(); }

  void *allocateMem(ContextImplPtr Context, bool InitFromUserData,
                    RT::PiEvent &OutEventToWait) override {
    void *UserPtr = InitFromUserData ? BaseT::getUserPtr() : nullptr;

    RT::PiMemImageDesc Desc = getImageDesc(UserPtr != nullptr);
    assert(checkImageDesc(Desc, Context, UserPtr) &&
           "The check an image desc failed.");

    RT::PiMemImageFormat Format = getImageFormat();
    assert(checkImageFormat(Format, Context) &&
           "The check an image format failed.");

    return MemoryManager::allocateMemImage(
        std::move(Context), this, UserPtr, BaseT::MHostPtrReadOnly,
        BaseT::getSize(), Desc, Format, BaseT::MInteropEvent,
        BaseT::MInteropContext, OutEventToWait);
  }

  MemObjType getType() const override { return MemObjType::IMAGE; }

  // Returns a valid accessor to the image with the specified access mode and
  // target. The only valid types for dataT are cl_int4, cl_uint4, cl_float4 and
  // cl_half4.
  template <typename DataT, access::mode AccessMode,
            typename = EnableIfImgAccDataT<DataT>>
  accessor<DataT, Dimensions, AccessMode, access::target::image,
           access::placeholder::false_t>
  get_access(image<Dimensions, AllocatorT> &Image,
             handler &CommandGroupHandler) {
    return accessor<DataT, Dimensions, AccessMode, access::target::image,
                    access::placeholder::false_t>(Image, CommandGroupHandler);
  }

  // Returns a valid accessor to the image immediately on the host with the
  // specified access mode and target. The only valid types for dataT are
  // cl_int4, cl_uint4, cl_float4 and cl_half4.
  template <typename DataT,
            access::mode AccessMode> //, typename = EnableIfImgAccDataT<DataT>>
  accessor<DataT, Dimensions, AccessMode, access::target::host_image,
           access::placeholder::false_t>
  get_access(image<Dimensions, AllocatorT> &Image) {
    return accessor<DataT, Dimensions, AccessMode, access::target::host_image,
                    access::placeholder::false_t>(Image);
  }

  // This utility api is currently used by accessor to get the element size of
  // the image. Element size is dependent on num of channels and channel type.
  // This information is not accessible from the image using any public API.
  size_t getElementSize() const { return MElementSize; };

  image_channel_order getChannelOrder() const { return MOrder; }

  image_channel_type getChannelType() const { return MType; }

  size_t getRowPitch() const { return MRowPitch; }

  size_t getSlicePitch() const { return MSlicePitch; }

  ~image_impl() { BaseT::updateHostMemory(); }

private:
  template <typename T> void getImageInfo(RT::PiMemImageInfo Info, T &Dest) {
    RT::PiMem Mem = pi::cast<RT::PiMem>(BaseT::MInteropMemObject);
    PI_CALL(RT::piMemImageGetInfo(Mem, Info, sizeof(T), &Dest, nullptr));
  }

  template <info::device Param>
  bool checkImageValueRange(const ContextImplPtr Context, const size_t Value) {
    const auto &Devices = Context->get_devices();
    return Value >= 1 && std::all_of(Devices.cbegin(), Devices.cend(),
                                     [Value](const device &Dev) {
                                       return Value <= Dev.get_info<Param>();
                                     });
  }

  template <typename T, typename... Args> bool checkAnyImpl(T Value) {
    return false;
  }

  template <typename ValT, typename VarT, typename... Args>
  bool checkAnyImpl(ValT Value, VarT Variant, Args... Arguments) {
    return (Value == Variant) ? true : checkAnyImpl(Value, Arguments...);
  }

  template <typename T, typename... Args>
  bool checkAny(const T Value, Args... Arguments) {
    return checkAnyImpl(Value, Arguments...);
  }

  RT::PiMemObjectType getImageType() {
    if (Dimensions == 1)
      return (MIsArrayImage ? PI_MEM_TYPE_IMAGE1D_ARRAY : PI_MEM_TYPE_IMAGE1D);
    if (Dimensions == 2)
      return (MIsArrayImage ? PI_MEM_TYPE_IMAGE2D_ARRAY : PI_MEM_TYPE_IMAGE2D);
    return PI_MEM_TYPE_IMAGE3D;
  }

  RT::PiMemImageDesc getImageDesc(bool InitFromHostPtr) {
    RT::PiMemImageDesc Desc;
    Desc.image_type = getImageType();
    Desc.image_width = MRange[0];
    Desc.image_height = Dimensions > 1 ? MRange[1] : 1;
    Desc.image_depth = Dimensions > 2 ? MRange[2] : 1;
    // TODO handle cases with IMAGE1D_ARRAY and IMAGE2D_ARRAY
    Desc.image_array_size = 0;
    // Pitches must be 0 if host ptr is not provided.
    Desc.image_row_pitch = InitFromHostPtr ? MRowPitch : 0;
    Desc.image_slice_pitch = InitFromHostPtr ? MSlicePitch : 0;

    Desc.num_mip_levels = 0;
    Desc.num_samples = 0;
    Desc.buffer = nullptr;
    return Desc;
  }

  bool checkImageDesc(const RT::PiMemImageDesc &Desc, ContextImplPtr Context,
                      void *UserPtr) {
    if (checkAny(Desc.image_type, PI_MEM_TYPE_IMAGE1D,
                 PI_MEM_TYPE_IMAGE1D_ARRAY, PI_MEM_TYPE_IMAGE2D_ARRAY,
                 PI_MEM_TYPE_IMAGE2D) &&
        !checkImageValueRange<info::device::image2d_max_width>(
            Context, Desc.image_width))
      throw invalid_parameter_error(
          "For a 1D/2D image/image array, the width must be a Value >= 1 and "
          "<= CL_DEVICE_IMAGE2D_MAX_WIDTH.");

    if (checkAny(Desc.image_type, PI_MEM_TYPE_IMAGE3D) &&
        !checkImageValueRange<info::device::image3d_max_width>(
            Context, Desc.image_width))
      throw invalid_parameter_error(
          "For a 3D image, the width must be a Value >= 1 and <= "
          "CL_DEVICE_IMAGE3D_MAX_WIDTH");

    if (checkAny(Desc.image_type, PI_MEM_TYPE_IMAGE2D,
                 PI_MEM_TYPE_IMAGE2D_ARRAY) &&
        !checkImageValueRange<info::device::image2d_max_height>(
            Context, Desc.image_height))
      throw invalid_parameter_error("For a 2D image or image array, the height "
                                    "must be a Value >= 1 and <= "
                                    "CL_DEVICE_IMAGE2D_MAX_HEIGHT");

    if (checkAny(Desc.image_type, PI_MEM_TYPE_IMAGE3D) &&
        !checkImageValueRange<info::device::image3d_max_height>(
            Context, Desc.image_height))
      throw invalid_parameter_error(
          "For a 3D image, the heightmust be a Value >= 1 and <= "
          "CL_DEVICE_IMAGE3D_MAX_HEIGHT");

    if (checkAny(Desc.image_type, PI_MEM_TYPE_IMAGE3D) &&
        !checkImageValueRange<info::device::image3d_max_depth>(
            Context, Desc.image_depth))
      throw invalid_parameter_error(
          "For a 3D image, the depth must be a Value >= 1 and <= "
          "CL_DEVICE_IMAGE3D_MAX_DEPTH");

    if (checkAny(Desc.image_type, PI_MEM_TYPE_IMAGE1D_ARRAY,
                 PI_MEM_TYPE_IMAGE2D_ARRAY) &&
        !checkImageValueRange<info::device::image_max_array_size>(
            Context, Desc.image_array_size))
      throw invalid_parameter_error(
          "For a 1D and 2D image array, the array_size must be a "
          "Value >= 1 and <= "
          "CL_DEVICE_IMAGE_MAX_ARRAY_SIZE.");

    if ((nullptr == UserPtr) && (0 != Desc.image_row_pitch))
      throw invalid_parameter_error(
          "The row_pitch must be 0 if host_ptr is nullptr.");

    if ((nullptr == UserPtr) && (0 != Desc.image_slice_pitch))
      throw invalid_parameter_error(
          "The slice_pitch must be 0 if host_ptr is nullptr.");

    if (0 != Desc.num_mip_levels)
      throw invalid_parameter_error("The mip_levels must be 0.");

    if (0 != Desc.num_samples)
      throw invalid_parameter_error("The num_samples must be 0.");

    if (nullptr != Desc.buffer)
      throw invalid_parameter_error(
          "The buffer must be nullptr, because SYCL does not support "
          "image creation from memory objects.");

    return true;
  }

  RT::PiMemImageFormat getImageFormat() {
    RT::PiMemImageFormat Format;
    Format.image_channel_order = detail::convertChannelOrder(MOrder);
    Format.image_channel_data_type = detail::convertChannelType(MType);
    return Format;
  }

  bool checkImageFormat(const RT::PiMemImageFormat &Format,
                        ContextImplPtr Context) {
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
          "CL_SNORM_INT16, CL_HALF_FLOAT, or CL_FLOAT. ");

    if (checkAny(Format.image_channel_order, PI_IMAGE_CHANNEL_ORDER_RGB,
                 PI_IMAGE_CHANNEL_ORDER_RGBx) &&
        !checkAny(Format.image_channel_data_type,
                  PI_IMAGE_CHANNEL_TYPE_UNORM_SHORT_565,
                  PI_IMAGE_CHANNEL_TYPE_UNORM_SHORT_555,
                  PI_IMAGE_CHANNEL_TYPE_UNORM_INT_101010))
      throw invalid_parameter_error(
          "CL_RGB or CL_RGBx	These formats can only be used if channel data "
          "type = CL_UNORM_SHORT_565, CL_UNORM_SHORT_555 or "
          "CL_UNORM_INT_101010. ");

    if (checkAny(Format.image_channel_order, PI_IMAGE_CHANNEL_ORDER_ARGB,
                 PI_IMAGE_CHANNEL_ORDER_BGRA, PI_IMAGE_CHANNEL_ORDER_ABGR) &&
        !checkAny(
            Format.image_channel_data_type, PI_IMAGE_CHANNEL_TYPE_UNORM_INT8,
            PI_IMAGE_CHANNEL_TYPE_SNORM_INT8, PI_IMAGE_CHANNEL_TYPE_SIGNED_INT8,
            PI_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8))
      throw invalid_parameter_error(
          "CL_ARGB, CL_BGRA, CL_ABGR	These formats can only be used if "
          "channel data type = CL_UNORM_INT8, CL_SNORM_INT8, CL_SIGNED_INT8 "
          "or CL_UNSIGNED_INT8.");

    return true;
  }

  bool MIsArrayImage = false;
  range<Dimensions> MRange;
  image_channel_order MOrder;
  image_channel_type MType;
  uint8_t MNumChannels = 0; // Maximum Value - 4
  size_t MElementSize = 0; // Maximum Value - 16
  size_t MRowPitch = 0;
  size_t MSlicePitch = 0;
};

} // namespace detail
} // namespace sycl
} // namespace cl
