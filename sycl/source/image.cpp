#include <detail/image_impl.hpp>
#include <sycl/event.hpp>
#include <sycl/ext/oneapi/accessor_property_list.hpp>
#include <sycl/image.hpp>
#include <sycl/properties/all_properties.hpp>
#include <sycl/property_list.hpp>
#include <sycl/range.hpp>

#include <memory>

namespace sycl {
inline namespace _V1 {
namespace detail {

image_plain::image_plain(image_channel_order Order, image_channel_type Type,
                         const range<3> &Range,
                         std::unique_ptr<SYCLMemObjAllocator> Allocator,
                         uint8_t Dimensions, const property_list &PropList) {
  impl = std::make_shared<detail::image_impl>(
      Order, Type, Range, std::move(Allocator), Dimensions, PropList);
}

image_plain::image_plain(image_channel_order Order, image_channel_type Type,
                         const range<3> &Range, const range<2> &Pitch,
                         std::unique_ptr<SYCLMemObjAllocator> Allocator,
                         uint8_t Dimensions, const property_list &PropList) {
  impl = std::make_shared<detail::image_impl>(
      Order, Type, Range, Pitch, std::move(Allocator), Dimensions, PropList);
}

image_plain::image_plain(void *HostPointer, image_channel_order Order,
                         image_channel_type Type, const range<3> &Range,
                         std::unique_ptr<SYCLMemObjAllocator> Allocator,
                         uint8_t Dimensions, const property_list &PropList) {
  impl = std::make_shared<detail::image_impl>(HostPointer, Order, Type, Range,
                                              std::move(Allocator), Dimensions,
                                              PropList);
}

image_plain::image_plain(const void *HostPointer, image_channel_order Order,
                         image_channel_type Type, const range<3> &Range,
                         std::unique_ptr<SYCLMemObjAllocator> Allocator,
                         uint8_t Dimensions, const property_list &PropList) {
  impl = std::make_shared<detail::image_impl>(HostPointer, Order, Type, Range,
                                              std::move(Allocator), Dimensions,
                                              PropList);
}

image_plain::image_plain(void *HostPointer, image_channel_order Order,
                         image_channel_type Type, const range<3> &Range,
                         const range<2> &Pitch,
                         std::unique_ptr<SYCLMemObjAllocator> Allocator,
                         uint8_t Dimensions, const property_list &PropList) {
  impl = std::make_shared<detail::image_impl>(HostPointer, Order, Type, Range,
                                              Pitch, std::move(Allocator),
                                              Dimensions, PropList);
}

image_plain::image_plain(const std::shared_ptr<const void> &HostPointer,
                         image_channel_order Order, image_channel_type Type,
                         const range<3> &Range,
                         std::unique_ptr<SYCLMemObjAllocator> Allocator,
                         uint8_t Dimensions, const property_list &PropList,
                         bool IsConstPtr) {
  impl = std::make_shared<detail::image_impl>(HostPointer, Order, Type, Range,
                                              std::move(Allocator), Dimensions,
                                              PropList, IsConstPtr);
}

image_plain::image_plain(const std::shared_ptr<const void> &HostPointer,
                         image_channel_order Order, image_channel_type Type,
                         const range<3> &Range, const range<2> &Pitch,
                         std::unique_ptr<SYCLMemObjAllocator> Allocator,
                         uint8_t Dimensions, const property_list &PropList,
                         bool IsConstPtr) {
  impl = std::make_shared<detail::image_impl>(HostPointer, Order, Type, Range,
                                              Pitch, std::move(Allocator),
                                              Dimensions, PropList, IsConstPtr);
}

image_plain::image_plain(const void *HostPointer, image_channel_order Order,
                         image_channel_type Type, image_sampler Sampler,
                         const range<3> &Range,
                         std::unique_ptr<SYCLMemObjAllocator> Allocator,
                         uint8_t Dimensions, const property_list &PropList) {
  impl = std::make_shared<detail::image_impl>(HostPointer, Order, Type, Sampler,
                                              Range, std::move(Allocator),
                                              Dimensions, PropList);
}

image_plain::image_plain(const void *HostPointer, image_channel_order Order,
                         image_channel_type Type, image_sampler Sampler,
                         const range<3> &Range, const range<2> &Pitch,
                         std::unique_ptr<SYCLMemObjAllocator> Allocator,
                         uint8_t Dimensions, const property_list &PropList) {
  impl = std::make_shared<detail::image_impl>(
      HostPointer, Order, Type, Sampler, Range, Pitch, std::move(Allocator),
      Dimensions, PropList);
}

image_plain::image_plain(const std::shared_ptr<const void> &HostPointer,
                         image_channel_order Order, image_channel_type Type,
                         image_sampler Sampler, const range<3> &Range,
                         std::unique_ptr<SYCLMemObjAllocator> Allocator,
                         uint8_t Dimensions, const property_list &PropList) {
  impl = std::make_shared<detail::image_impl>(HostPointer, Order, Type, Sampler,
                                              Range, std::move(Allocator),
                                              Dimensions, PropList);
}

image_plain::image_plain(const std::shared_ptr<const void> &HostPointer,
                         image_channel_order Order, image_channel_type Type,
                         image_sampler Sampler, const range<3> &Range,
                         const range<2> &Pitch,
                         std::unique_ptr<SYCLMemObjAllocator> Allocator,
                         uint8_t Dimensions, const property_list &PropList) {
  impl = std::make_shared<detail::image_impl>(
      HostPointer, Order, Type, Sampler, Range, Pitch, std::move(Allocator),
      Dimensions, PropList);
}

#ifdef __SYCL_INTERNAL_API
image_plain::image_plain(cl_mem ClMemObject, const context &SyclContext,
                         event AvailableEvent,
                         std::unique_ptr<SYCLMemObjAllocator> Allocator,
                         uint8_t Dimensions) {
  impl = std::make_shared<detail::image_impl>(ClMemObject, SyclContext,
                                              AvailableEvent,
                                              std::move(Allocator), Dimensions);
}
#endif

image_plain::image_plain(pi_native_handle MemObject, const context &SyclContext,
                         event AvailableEvent,
                         std::unique_ptr<SYCLMemObjAllocator> Allocator,
                         uint8_t Dimensions, image_channel_order Order,
                         image_channel_type Type, bool OwnNativeHandle,
                         range<3> Range3WithOnes) {
  impl = std::make_shared<detail::image_impl>(
      MemObject, SyclContext, AvailableEvent, std::move(Allocator), Dimensions,
      Order, Type, OwnNativeHandle, Range3WithOnes);
}

range<3> image_plain::get_range() const { return impl->get_range(); }

range<2> image_plain::get_pitch() const { return impl->get_pitch(); }

size_t image_plain::get_size() const noexcept { return impl->getSizeInBytes(); }

size_t image_plain::get_count() const noexcept { return impl->get_count(); }

void image_plain::set_final_data_internal() { impl->set_final_data(nullptr); }

void image_plain::set_final_data_internal(
    const std::function<void(const std::function<void(void *const Ptr)> &)>
        &FinalDataFunc) {
  impl->set_final_data(FinalDataFunc);
}

void image_plain::set_write_back(bool NeedWriteBack) {
  impl->set_write_back(NeedWriteBack);
}

const std::unique_ptr<SYCLMemObjAllocator> &
image_plain::get_allocator_internal() const {
  return impl->get_allocator_internal();
}

size_t image_plain::getElementSize() const { return impl->getElementSize(); }

size_t image_plain::getRowPitch() const { return impl->getRowPitch(); }

size_t image_plain::getSlicePitch() const { return impl->getSlicePitch(); }

image_sampler image_plain::getSampler() const noexcept {
  return impl->getSampler();
}

image_channel_order image_plain::getChannelOrder() const {
  return impl->getChannelOrder();
}

image_channel_type image_plain::getChannelType() const {
  return impl->getChannelType();
}

void image_plain::sampledImageConstructorNotification(
    const detail::code_location &CodeLoc, void *UserObj, const void *HostObj,
    uint32_t Dim, size_t Range[3], image_format Format,
    const image_sampler &Sampler) {
  impl->sampledImageConstructorNotification(CodeLoc, UserObj, HostObj, Dim,
                                            Range, Format, Sampler);
}

void image_plain::sampledImageDestructorNotification(void *UserObj) {
  impl->sampledImageDestructorNotification(UserObj);
}

void image_plain::unsampledImageConstructorNotification(
    const detail::code_location &CodeLoc, void *UserObj, const void *HostObj,
    uint32_t Dim, size_t Range[3], image_format Format) {
  impl->unsampledImageConstructorNotification(CodeLoc, UserObj, HostObj, Dim,
                                              Range, Format);
}

void image_plain::unsampledImageDestructorNotification(void *UserObj) {
  impl->unsampledImageDestructorNotification(UserObj);
}

const property_list &image_plain::getPropList() const {
  return impl->getPropList();
}

} // namespace detail
} // namespace _V1
} // namespace sycl
