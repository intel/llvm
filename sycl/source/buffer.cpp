#include <detail/buffer_impl.hpp>
#include <sycl/buffer.hpp>
#include <sycl/ext/oneapi/accessor_property_list.hpp>
#include <sycl/properties/all_properties.hpp>
#include <sycl/property_list.hpp>
#include <sycl/range.hpp>

#include <memory>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {

buffer_plain::buffer_plain(
    size_t SizeInBytes, size_t RequiredAlign, const property_list &Props,
    std::unique_ptr<detail::SYCLMemObjAllocator> Allocator) {

  impl = std::make_shared<detail::buffer_impl>(SizeInBytes, RequiredAlign,
                                               Props, std::move(Allocator));
}

buffer_plain::buffer_plain(
    void *HostData, size_t SizeInBytes, size_t RequiredAlign,
    const sycl::property_list &Props,
    std::unique_ptr<sycl::detail::SYCLMemObjAllocator> Allocator) {
  impl = std::make_shared<detail::buffer_impl>(
      HostData, SizeInBytes, RequiredAlign, Props, std::move(Allocator));
}

buffer_plain::buffer_plain(
    const void *HostData, size_t SizeInBytes, size_t RequiredAlign,
    const property_list &Props,
    std::unique_ptr<detail::SYCLMemObjAllocator> Allocator) {
  impl = std::make_shared<detail::buffer_impl>(
      HostData, SizeInBytes, RequiredAlign, Props, std::move(Allocator));
}

buffer_plain::buffer_plain(
    const std::shared_ptr<const void> &HostData, const size_t SizeInBytes,
    size_t RequiredAlign, const property_list &Props,
    std::unique_ptr<detail::SYCLMemObjAllocator> Allocator, bool IsConstPtr) {
  impl = std::make_shared<detail::buffer_impl>(
      HostData, SizeInBytes, RequiredAlign, Props, std::move(Allocator),
      IsConstPtr);
}

buffer_plain::buffer_plain(
    const std::function<void(void *)> &CopyFromInput, const size_t SizeInBytes,
    size_t RequiredAlign, const property_list &Props,
    std::unique_ptr<detail::SYCLMemObjAllocator> Allocator, bool IsConstPtr) {
  impl = std::make_shared<detail::buffer_impl>(
      CopyFromInput, SizeInBytes, RequiredAlign, Props, std::move(Allocator),
      IsConstPtr);
}

buffer_plain::buffer_plain(
    pi_native_handle MemObject, context SyclContext,
    std::unique_ptr<detail::SYCLMemObjAllocator> Allocator,
    bool OwnNativeHandle, event AvailableEvent) {
  impl = std::make_shared<detail::buffer_impl>(
      MemObject, std::move(SyclContext), std::move(Allocator), OwnNativeHandle,
      std::move(AvailableEvent));
}

void buffer_plain::set_final_data_internal() { impl->set_final_data(nullptr); }

void buffer_plain::set_final_data_internal(
    const std::function<void(const std::function<void(void *const Ptr)> &)>
        &FinalDataFunc) {
  impl->set_final_data(FinalDataFunc);
}

void buffer_plain::constructorNotification(const detail::code_location &CodeLoc,
                                           void *UserObj, const void *HostObj,
                                           const void *Type, uint32_t Dim,
                                           uint32_t ElemType, size_t Range[3]) {
  impl->constructorNotification(CodeLoc, UserObj, HostObj, Type, Dim, ElemType,
                                Range);
}

void buffer_plain::set_write_back(bool NeedWriteBack) {
  impl->set_write_back(NeedWriteBack);
}

#define __SYCL_PARAM_TRAITS_SPEC(param_type)                                   \
  template <>                                                                  \
  __SYCL_EXPORT bool buffer_plain::has_property<param_type>() const noexcept { \
    return impl->has_property<param_type>();                                   \
  }
#include <sycl/detail/properties_traits.def>

#undef __SYCL_PARAM_TRAITS_SPEC

#define __SYCL_PARAM_TRAITS_SPEC(param_type)                                   \
  template <>                                                                  \
  __SYCL_EXPORT param_type buffer_plain::get_property<param_type>() const {    \
    return impl->get_property<param_type>();                                   \
  }
#include <sycl/detail/properties_traits.def>

#undef __SYCL_PARAM_TRAITS_SPEC

std::vector<pi_native_handle>
buffer_plain::getNativeVector(backend BackendName) const {
  return impl->getNativeVector(BackendName);
}

const std::unique_ptr<SYCLMemObjAllocator> &
buffer_plain::get_allocator_internal() const {
  return impl->get_allocator_internal();
}

void buffer_plain::deleteAccProps(const sycl::detail::PropWithDataKind &Kind) {
  impl->deleteAccessorProperty(Kind);
}

void buffer_plain::addOrReplaceAccessorProperties(
    const property_list &PropertyList) {
  impl->addOrReplaceAccessorProperties(PropertyList);
}

size_t buffer_plain::getSize() const { return impl->getSizeInBytes(); }

void buffer_plain::handleRelease() const {
  // Try to detach memory object only if impl is going to be released.
  // Buffer copy will have pointer to the same impl.
  if (impl.use_count() == 1)
    impl->detachMemoryObject(impl);
}

} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
