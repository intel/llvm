//==------------ image_impl.hpp --------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <detail/sycl_mem_obj_t.hpp>
#include <sycl/detail/aligned_allocator.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/detail/export.hpp>
#include <sycl/detail/generic_type_traits.hpp>
#include <sycl/device.hpp>
#include <sycl/event.hpp>
#include <sycl/image.hpp>
#include <sycl/property_list.hpp>
#include <sycl/range.hpp>
#include <sycl/sampler.hpp>

namespace sycl {
inline namespace _V1 {

// forward declarations
enum class image_channel_order : unsigned int;
enum class image_channel_type : unsigned int;

template <int Dimensions, typename AllocatorT> class image;
template <typename DataT, int Dimensions, access::mode AccessMode,
          access::target AccessTarget, access::placeholder IsPlaceholder,
          typename property_listT>
class accessor;
class handler;

namespace detail {

// utility functions and typedefs for image_impl
using image_allocator = aligned_allocator<byte>;

// utility function: Returns the Number of Channels for a given Order.
uint8_t getImageNumberChannels(image_channel_order Order);

// utility function: Returns the number of bytes per image element
uint8_t getImageElementSize(uint8_t NumChannels, image_channel_type Type);

sycl::detail::pi::PiMemImageChannelOrder
convertChannelOrder(image_channel_order Order);

image_channel_order
convertChannelOrder(sycl::detail::pi::PiMemImageChannelOrder Order);

sycl::detail::pi::PiMemImageChannelType
convertChannelType(image_channel_type Type);

image_channel_type
convertChannelType(sycl::detail::pi::PiMemImageChannelType Type);

class image_impl final : public SYCLMemObjT {
  using BaseT = SYCLMemObjT;
  using typename BaseT::MemObjType;

private:
  void setPitches() {
    size_t WHD[3] = {1, 1, 1}; // Width, Height, Depth.
    for (int I = 0; I < MDimensions; I++)
      WHD[I] = MRange[I];

    MRowPitch = MElementSize * WHD[0];
    MSlicePitch = MRowPitch * WHD[1];
    BaseT::MSizeInBytes = MSlicePitch * WHD[2];
  }

  void setPitches(const range<2> &Pitch) {
    MRowPitch = Pitch[0];
    MSlicePitch =
        (MDimensions == 3) ? Pitch[1] : MRowPitch; // Dimensions will be 2/3.
    // NumSlices is depth when dim==3, and height when dim==2.
    size_t NumSlices =
        (MDimensions == 3) ? MRange[2] : MRange[1]; // Dimensions will be 2/3.

    BaseT::MSizeInBytes = MSlicePitch * NumSlices;
  }

public:
  image_impl(image_channel_order Order, image_channel_type Type,
             const range<3> &ImageRange,
             std::unique_ptr<SYCLMemObjAllocator> Allocator, uint8_t Dimensions,
             const property_list &PropList = {})
      : image_impl((void *)nullptr, Order, Type, ImageRange,
                   std::move(Allocator), Dimensions, PropList) {}

  image_impl(image_channel_order Order, image_channel_type Type,
             const range<3> &ImageRange, const range<2> &Pitch,
             std::unique_ptr<SYCLMemObjAllocator> Allocator, uint8_t Dimensions,
             const property_list &PropList = {})
      : image_impl((void *)nullptr, Order, Type, ImageRange, Pitch,
                   std::move(Allocator), Dimensions, PropList) {}

  image_impl(void *HData, image_channel_order Order, image_channel_type Type,
             const range<3> &ImageRange,
             std::unique_ptr<SYCLMemObjAllocator> Allocator, uint8_t Dimensions,
             const property_list &PropList = {})
      : BaseT(PropList, std::move(Allocator)), MDimensions(Dimensions),
        MRange(ImageRange), MOrder(Order), MType(Type),
        MNumChannels(getImageNumberChannels(MOrder)),
        MElementSize(getImageElementSize(MNumChannels, MType)) {
    setPitches();
    BaseT::handleHostData(HData, detail::getNextPowerOfTwo(MElementSize));
  }

  image_impl(const void *HData, image_channel_order Order,
             image_channel_type Type, const range<3> &ImageRange,
             std::unique_ptr<SYCLMemObjAllocator> Allocator, uint8_t Dimensions,
             const property_list &PropList = {})
      : BaseT(PropList, std::move(Allocator)), MDimensions(Dimensions),
        MRange(ImageRange), MOrder(Order), MType(Type),
        MNumChannels(getImageNumberChannels(MOrder)),
        MElementSize(getImageElementSize(MNumChannels, MType)) {
    setPitches();
    BaseT::handleHostData(HData, detail::getNextPowerOfTwo(MElementSize));
  }

  image_impl(void *HData, image_channel_order Order, image_channel_type Type,
             const range<3> &ImageRange, const range<2> &Pitch,
             std::unique_ptr<SYCLMemObjAllocator> Allocator, uint8_t Dimensions,
             const property_list &PropList = {})
      : BaseT(PropList, std::move(Allocator)), MDimensions(Dimensions),
        MRange(ImageRange), MOrder(Order), MType(Type),
        MNumChannels(getImageNumberChannels(MOrder)),
        MElementSize(getImageElementSize(MNumChannels, MType)) {
    setPitches(Pitch);
    BaseT::handleHostData(HData, detail::getNextPowerOfTwo(MElementSize));
  }

  image_impl(const std::shared_ptr<const void> &HData,
             image_channel_order Order, image_channel_type Type,
             const range<3> &ImageRange,
             std::unique_ptr<SYCLMemObjAllocator> Allocator, uint8_t Dimensions,
             const property_list &PropList, bool IsConstPtr)
      : BaseT(PropList, std::move(Allocator)), MDimensions(Dimensions),
        MRange(ImageRange), MOrder(Order), MType(Type),
        MNumChannels(getImageNumberChannels(MOrder)),
        MElementSize(getImageElementSize(MNumChannels, MType)) {
    setPitches();
    BaseT::handleHostData(std::const_pointer_cast<void>(HData),
                          detail::getNextPowerOfTwo(MElementSize), IsConstPtr);
  }

  image_impl(const std::shared_ptr<const void> &HData,
             image_channel_order Order, image_channel_type Type,
             const range<3> &ImageRange, const range<2> &Pitch,
             std::unique_ptr<SYCLMemObjAllocator> Allocator, uint8_t Dimensions,
             const property_list &PropList, bool IsConstPtr)
      : BaseT(PropList, std::move(Allocator)), MDimensions(Dimensions),
        MRange(ImageRange), MOrder(Order), MType(Type),
        MNumChannels(getImageNumberChannels(MOrder)),
        MElementSize(getImageElementSize(MNumChannels, MType)) {
    setPitches(Pitch);
    BaseT::handleHostData(std::const_pointer_cast<void>(HData),
                          detail::getNextPowerOfTwo(MElementSize), IsConstPtr);
  }

  image_impl(const void *HData, image_channel_order Order,
             image_channel_type Type, image_sampler Sampler,
             const range<3> &ImageRange,
             std::unique_ptr<SYCLMemObjAllocator> Allocator, uint8_t Dimensions,
             const property_list &PropList = {})
      : BaseT(PropList, std::move(Allocator)), MDimensions(Dimensions),
        MRange(ImageRange), MOrder(Order), MType(Type),
        MNumChannels(getImageNumberChannels(MOrder)),
        MElementSize(getImageElementSize(MNumChannels, MType)),
        MSampler(Sampler) {
    setPitches();
    BaseT::handleHostData(HData, detail::getNextPowerOfTwo(MElementSize));
  }

  image_impl(const void *HData, image_channel_order Order,
             image_channel_type Type, image_sampler Sampler,
             const range<3> &ImageRange, const range<2> &Pitch,
             std::unique_ptr<SYCLMemObjAllocator> Allocator, uint8_t Dimensions,
             const property_list &PropList = {})
      : BaseT(PropList, std::move(Allocator)), MDimensions(Dimensions),
        MRange(ImageRange), MOrder(Order), MType(Type),
        MNumChannels(getImageNumberChannels(MOrder)),
        MElementSize(getImageElementSize(MNumChannels, MType)),
        MSampler(Sampler) {
    setPitches(Pitch);
    BaseT::handleHostData(HData, detail::getNextPowerOfTwo(MElementSize));
  }

  image_impl(const std::shared_ptr<const void> &HData,
             image_channel_order Order, image_channel_type Type,
             image_sampler Sampler, const range<3> &ImageRange,
             std::unique_ptr<SYCLMemObjAllocator> Allocator, uint8_t Dimensions,
             const property_list &PropList)
      : BaseT(PropList, std::move(Allocator)), MDimensions(Dimensions),
        MRange(ImageRange), MOrder(Order), MType(Type),
        MNumChannels(getImageNumberChannels(MOrder)),
        MElementSize(getImageElementSize(MNumChannels, MType)),
        MSampler(Sampler) {
    setPitches();
    BaseT::handleHostData(std::const_pointer_cast<void>(HData),
                          detail::getNextPowerOfTwo(MElementSize),
                          /*IsConstPtr=*/true);
  }

  image_impl(const std::shared_ptr<const void> &HData,
             image_channel_order Order, image_channel_type Type,
             image_sampler Sampler, const range<3> &ImageRange,
             const range<2> &Pitch,
             std::unique_ptr<SYCLMemObjAllocator> Allocator, uint8_t Dimensions,
             const property_list &PropList)
      : BaseT(PropList, std::move(Allocator)), MDimensions(Dimensions),
        MRange(ImageRange), MOrder(Order), MType(Type),
        MNumChannels(getImageNumberChannels(MOrder)),
        MElementSize(getImageElementSize(MNumChannels, MType)),
        MSampler(Sampler) {
    setPitches(Pitch);
    BaseT::handleHostData(std::const_pointer_cast<void>(HData),
                          detail::getNextPowerOfTwo(MElementSize),
                          /*IsConstPtr=*/true);
  }

  image_impl(cl_mem MemObject, const context &SyclContext, event AvailableEvent,
             std::unique_ptr<SYCLMemObjAllocator> Allocator,
             uint8_t Dimensions);

  image_impl(pi_native_handle MemObject, const context &SyclContext,
             event AvailableEvent,
             std::unique_ptr<SYCLMemObjAllocator> Allocator, uint8_t Dimensions,
             image_channel_order Order, image_channel_type Type,
             bool OwnNativeHandle, range<3> Range3WithOnes);

  // Return a range object representing the size of the image in terms of the
  // number of elements in each dimension as passed to the constructor
  range<3> get_range() const { return MRange; }

  // Return a range object representing the pitch of the image in bytes.
  range<2> get_pitch() const { return {MRowPitch, MSlicePitch}; }

  // Returns the total number of elements in the image
  size_t get_count() const { return size(); }
  size_t size() const noexcept { return MRange.size(); }

  void *allocateMem(ContextImplPtr Context, bool InitFromUserData,
                    void *HostPtr,
                    sycl::detail::pi::PiEvent &OutEventToWait) override;

  MemObjType getType() const override { return MemObjType::Image; }

  // This utility api is currently used by accessor to get the element size of
  // the image. Element size is dependent on num of channels and channel type.
  // This information is not accessible from the image using any public API.
  size_t getElementSize() const { return MElementSize; };

  image_channel_order getChannelOrder() const { return MOrder; }

  image_channel_type getChannelType() const { return MType; }

  size_t getRowPitch() const { return MRowPitch; }

  size_t getSlicePitch() const { return MSlicePitch; }

  image_sampler getSampler() const noexcept {
    return MSampler.value_or(image_sampler{
        addressing_mode::none, coordinate_normalization_mode::unnormalized,
        filtering_mode::linear});
  }

  ~image_impl() {
    try {
      BaseT::updateHostMemory();
    } catch (...) {
    }
  }

  void sampledImageConstructorNotification(const detail::code_location &CodeLoc,
                                           void *UserObj, const void *HostObj,
                                           uint32_t Dim, size_t Range[3],
                                           image_format Format,
                                           const image_sampler &Sampler);
  void sampledImageDestructorNotification(void *UserObj);

  void unsampledImageConstructorNotification(
      const detail::code_location &CodeLoc, void *UserObj, const void *HostObj,
      uint32_t Dim, size_t Range[3], image_format Format);
  void unsampledImageDestructorNotification(void *UserObj);

private:
  std::vector<device> getDevices(const ContextImplPtr Context);

  sycl::detail::pi::PiMemObjectType getImageType() {
    if (MDimensions == 1)
      return (MIsArrayImage ? PI_MEM_TYPE_IMAGE1D_ARRAY : PI_MEM_TYPE_IMAGE1D);
    if (MDimensions == 2)
      return (MIsArrayImage ? PI_MEM_TYPE_IMAGE2D_ARRAY : PI_MEM_TYPE_IMAGE2D);
    return PI_MEM_TYPE_IMAGE3D;
  }

  sycl::detail::pi::PiMemImageDesc getImageDesc(bool InitFromHostPtr) {
    sycl::detail::pi::PiMemImageDesc Desc;
    Desc.image_type = getImageType();

    // MRange<> is [width], [width,height], or [width,height,depth] (which
    // is different than MAccessRange, etc in bufffers)
    constexpr int XTermPos = 0, YTermPos = 1, ZTermPos = 2;
    Desc.image_width = MRange[XTermPos];
    Desc.image_height = MDimensions > 1 ? MRange[YTermPos] : 1;
    Desc.image_depth = MDimensions > 2 ? MRange[ZTermPos] : 1;

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

  bool checkImageDesc(const sycl::detail::pi::PiMemImageDesc &Desc,
                      ContextImplPtr Context, void *UserPtr);

  sycl::detail::pi::PiMemImageFormat getImageFormat() {
    sycl::detail::pi::PiMemImageFormat Format;
    Format.image_channel_order = detail::convertChannelOrder(MOrder);
    Format.image_channel_data_type = detail::convertChannelType(MType);
    return Format;
  }

  bool checkImageFormat(const sycl::detail::pi::PiMemImageFormat &Format,
                        ContextImplPtr Context);

  uint8_t MDimensions = 0;
  bool MIsArrayImage = false;
  range<3> MRange;
  image_channel_order MOrder;
  image_channel_type MType;
  uint8_t MNumChannels = 0; // Maximum Value - 4
  size_t MElementSize = 0;  // Maximum Value - 16
  size_t MRowPitch = 0;
  size_t MSlicePitch = 0;

  // Image may carry a 2020 sampler.
  std::optional<image_sampler> MSampler = std::nullopt;
};
} // namespace detail
} // namespace _V1
} // namespace sycl
