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
#include <CL/sycl/detail/sycl_mem_obj_t.hpp>
#include <CL/sycl/device.hpp>
#include <CL/sycl/event.hpp>
#include <CL/sycl/property_list.hpp>
#include <CL/sycl/range.hpp>
#include <CL/sycl/stl.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
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

template <int Dimensions> class image_impl final : public SYCLMemObjT {
  using BaseT = SYCLMemObjT;
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
             unique_ptr_class<SYCLMemObjAllocator> Allocator,
             const property_list &PropList = {})
      : image_impl((void *)nullptr, Order, Type, ImageRange,
                   std::move(Allocator), PropList) {}

  template <bool B = (Dimensions > 1)>
  image_impl(image_channel_order Order, image_channel_type Type,
             const range<Dimensions> &ImageRange,
             const EnableIfPitchT<B> &Pitch,
             unique_ptr_class<SYCLMemObjAllocator> Allocator,
             const property_list &PropList = {})
      : image_impl((void *)nullptr, Order, Type, ImageRange, Pitch,
                   std::move(Allocator), PropList) {}

  image_impl(void *HData, image_channel_order Order, image_channel_type Type,
             const range<Dimensions> &ImageRange,
             unique_ptr_class<SYCLMemObjAllocator> Allocator,
             const property_list &PropList = {})
      : BaseT(PropList, std::move(Allocator)), MRange(ImageRange),
        MOrder(Order), MType(Type),
        MNumChannels(getImageNumberChannels(MOrder)),
        MElementSize(getImageElementSize(MNumChannels, MType)) {
    setPitches();
    BaseT::handleHostData(HData, detail::getNextPowerOfTwo(MElementSize));
  }

  image_impl(const void *HData, image_channel_order Order,
             image_channel_type Type, const range<Dimensions> &ImageRange,
             unique_ptr_class<SYCLMemObjAllocator> Allocator,
             const property_list &PropList = {})
      : BaseT(PropList, std::move(Allocator)), MRange(ImageRange),
        MOrder(Order), MType(Type),
        MNumChannels(getImageNumberChannels(MOrder)),
        MElementSize(getImageElementSize(MNumChannels, MType)) {
    setPitches();
    BaseT::handleHostData(HData, detail::getNextPowerOfTwo(MElementSize));
  }

  template <bool B = (Dimensions > 1)>
  image_impl(void *HData, image_channel_order Order, image_channel_type Type,
             const range<Dimensions> &ImageRange,
             const EnableIfPitchT<B> &Pitch,
             unique_ptr_class<SYCLMemObjAllocator> Allocator,
             const property_list &PropList = {})
      : BaseT(PropList, std::move(Allocator)), MRange(ImageRange),
        MOrder(Order), MType(Type),
        MNumChannels(getImageNumberChannels(MOrder)),
        MElementSize(getImageElementSize(MNumChannels, MType)) {
    setPitches(Pitch);
    BaseT::handleHostData(HData, detail::getNextPowerOfTwo(MElementSize));
  }

  image_impl(shared_ptr_class<void> &HData, image_channel_order Order,
             image_channel_type Type, const range<Dimensions> &ImageRange,
             unique_ptr_class<SYCLMemObjAllocator> Allocator,
             const property_list &PropList = {})
      : BaseT(PropList, std::move(Allocator)), MRange(ImageRange),
        MOrder(Order), MType(Type),
        MNumChannels(getImageNumberChannels(MOrder)),
        MElementSize(getImageElementSize(MNumChannels, MType)) {
    setPitches();
    BaseT::handleHostData(HData, detail::getNextPowerOfTwo(MElementSize));
  }

  /* Available only when: Dimensions > 1 */
  template <bool B = (Dimensions > 1)>
  image_impl(shared_ptr_class<void> &HData, image_channel_order Order,
             image_channel_type Type, const range<Dimensions> &ImageRange,
             const EnableIfPitchT<B> &Pitch,
             unique_ptr_class<SYCLMemObjAllocator> Allocator,
             const property_list &PropList = {})
      : BaseT(PropList, std::move(Allocator)), MRange(ImageRange),
        MOrder(Order), MType(Type),
        MNumChannels(getImageNumberChannels(MOrder)),
        MElementSize(getImageElementSize(MNumChannels, MType)) {
    setPitches(Pitch);
    BaseT::handleHostData(HData, detail::getNextPowerOfTwo(MElementSize));
  }

  image_impl(cl_mem MemObject, const context &SyclContext, event AvailableEvent,
             unique_ptr_class<SYCLMemObjAllocator> Allocator);

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
                    void *HostPtr, RT::PiEvent &OutEventToWait) override;

  MemObjType getType() const override { return MemObjType::IMAGE; }

  // This utility api is currently used by accessor to get the element size of
  // the image. Element size is dependent on num of channels and channel type.
  // This information is not accessible from the image using any public API.
  size_t getElementSize() const { return MElementSize; };

  image_channel_order getChannelOrder() const { return MOrder; }

  image_channel_type getChannelType() const { return MType; }

  size_t getRowPitch() const { return MRowPitch; }

  size_t getSlicePitch() const { return MSlicePitch; }

  ~image_impl() {
    try {
      BaseT::updateHostMemory();
    } catch (...) {
    }
  }

private:
  vector_class<device> getDevices(const ContextImplPtr Context);

  template <info::device Param>
  bool checkImageValueRange(const ContextImplPtr Context, const size_t Value) {
    const auto &Devices = getDevices(Context);
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
                      void *UserPtr);

  RT::PiMemImageFormat getImageFormat() {
    RT::PiMemImageFormat Format;
    Format.image_channel_order = detail::convertChannelOrder(MOrder);
    Format.image_channel_data_type = detail::convertChannelType(MType);
    return Format;
  }

  bool checkImageFormat(const RT::PiMemImageFormat &Format,
                        ContextImplPtr Context);

  bool MIsArrayImage = false;
  range<Dimensions> MRange;
  image_channel_order MOrder;
  image_channel_type MType;
  uint8_t MNumChannels = 0; // Maximum Value - 4
  size_t MElementSize = 0;  // Maximum Value - 16
  size_t MRowPitch = 0;
  size_t MSlicePitch = 0;
};
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
