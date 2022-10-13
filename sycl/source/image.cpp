#include <detail/image_impl.hpp>
#include <sycl/event.hpp>
#include <sycl/ext/oneapi/accessor_property_list.hpp>
#include <sycl/image.hpp>
#include <sycl/properties/all_properties.hpp>
#include <sycl/property_list.hpp>
#include <sycl/range.hpp>

#include <memory>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
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

#define __SYCL_PARAM_TRAITS_SPEC(param_type)                                   \
  template <>                                                                  \
  __SYCL_EXPORT bool image_plain::has_property<param_type>() const noexcept {  \
    return impl->has_property<param_type>();                                   \
  }
#include <sycl/detail/properties_traits.def>

#undef __SYCL_PARAM_TRAITS_SPEC

#define __SYCL_PARAM_TRAITS_SPEC(param_type)                                   \
  template <>                                                                  \
  __SYCL_EXPORT param_type image_plain::get_property<param_type>() const {     \
    return impl->get_property<param_type>();                                   \
  }
#include <sycl/detail/properties_traits.def>

#undef __SYCL_PARAM_TRAITS_SPEC

range<3> image_plain::get_range() const { return impl->get_range(); }

range<2> image_plain::get_pitch() const { return impl->get_pitch(); }

size_t image_plain::get_size() const { return impl->getSizeInBytes(); }

size_t image_plain::get_count() const { return impl->get_count(); }

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

image_channel_order image_plain::getChannelOrder() const {
  return impl->getChannelOrder();
}

image_channel_type image_plain::getChannelType() const {
  return impl->getChannelType();
}

} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
