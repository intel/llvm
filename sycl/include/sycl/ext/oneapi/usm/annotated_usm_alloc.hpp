#include <sycl/detail/usm_impl.hpp>
#include <sycl/ext/oneapi/annotated_arg/annotated_ptr.hpp>
#include <sycl/sycl.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext {
namespace oneapi {
namespace experimental {

using alloc = sycl::usm::alloc;

namespace {

template <typename T, typename propertyList>
annotated_ptr<T, propertyList>
annotated_ptr_cast(annotated_ptr<void, propertyList> annPtr) {
  return {annPtr.get()};
}

} // anonymous namespace

////
//  Device USM allocation functions with properties support
////

template <typename propertyListA, typename propertyListB>
annotated_ptr<void, propertyListB>
malloc_device_annotated(size_t numBytes, const device &syclDevice,
                        const context &syclContext,
                        const propertyListA &propList = properties{});

template <typename T, typename propertyListA, typename propertyListB>
annotated_ptr<T, propertyListB>
malloc_device_annotated(size_t count, const device &syclDevice,
                        const context &syclContext,
                        const propertyListA &propList = properties{}) {

  return {malloc_device_annotated<propertyListA, propertyListB>(
              align, count, syclDevice, syclContext, propList)
              .get()};
}

template <typename propertyListA, typename propertyListB>
annotated_ptr<void, propertyListB>
malloc_device_annotated(size_t numBytes, const queue &syclQueue,
                        const propertyListA &propList = properties{}) {
  return malloc_device_annotated<propertyListA, propertyListB>(
      numBytes, syclQueue.get_device(), syclQueue.get_context(), propList);
}

template <typename T, typename propertyListA, typename propertyListB>
annotated_ptr<T, propertyListB>
malloc_device_annotated(size_t count, const queue &syclQueue,
                        const propertyListA &propList = properties{}) {

  return malloc_device_annotated<T, propertyListA, propertyListB>(
      count, syclQueue.get_device(), syclQueue.get_context(), propList);
}

template <typename propertyListA, typename propertyListB>
annotated_ptr<void, propertyListB> aligned_alloc_device_annotated(
    size_t alignment, size_t numBytes, const device &syclDevice,
    const context &syclContext, const propertyListA &propList = properties{});

template <typename T, typename propertyListA, typename propertyListB>
annotated_ptr<T, propertyListB> aligned_alloc_device_annotated(
    size_t alignment, size_t count, const device &syclDevice,
    const context &syclContext, const propertyListA &propList = properties{}) {
  return {aligned_alloc_device_annotated<propertyListA, propertyListB>(
              align, count, syclDevice, syclContext, propList)
              .get()};
}

template <typename propertyListA, typename propertyListB>
annotated_ptr<void, propertyListB>
aligned_alloc_device_annotated(size_t alignment, size_t numBytes,
                               const queue &syclQueue,
                               const propertyListA &propList = properties{}) {
  return aligned_alloc_device_annotated<propertyListA, propertyListB>(
      alignment, numBytes, syclQueue.get_device(), syclQueue.get_context(),
      propList);
}

template <typename T, typename propertyListA, typename propertyListB>
annotated_ptr<T, propertyListB>
aligned_alloc_device_annotated(size_t alignment, size_t count,
                               const queue &syclQueue,
                               const propertyListA &propList = properties{}) {
  return aligned_alloc_device_annotated<T, propertyListA, propertyListB>(
      alignment, count, syclQueue.get_device(), syclQueue.get_context(),
      propList);
}

////
//  Host USM allocation functions with properties support
//
////

template <typename propertyListA, typename propertyListB>
annotated_ptr<void, propertyListB>
malloc_host_annotated(size_t numBytes, const context &syclContext,
                      const propertyListA &propList = properties{});

template <typename T, typename propertyListA, typename propertyListB>
annotated_ptr<T, propertyListB>
malloc_host_annotated(size_t count, const context &syclContext,
                      const propertyListA &propList = properties{}) {
  return {aligned_alloc_host_annotated<propertyListA, propertyListB>(
              count * sizeof(T), syclContext, propList)
              .get()};
}

template <typename propertyListA, typename propertyListB>
annotated_ptr<void, propertyListB>
malloc_host_annotated(size_t numBytes, const queue &syclQueue,
                      const propertyListA &propList = properties{}) {
  return malloc_host_annotated<propertyListA, propertyListB>(
      numBytes, syclQueue.get_context(), propList);
}

template <typename T, typename propertyListA, typename propertyListB>
annotated_ptr<T, propertyListB>
malloc_host_annotated(size_t count, const queue &syclQueue,
                      const propertyListA &propList = properties{}) {
  return malloc_host_annotated<T, propertyListA, propertyListB>(
      count, syclQueue.get_context(), propList);
}

template <typename propertyListA, typename propertyListB>
annotated_ptr<void, propertyListB>
aligned_alloc_host_annotated(size_t alignment, size_t numBytes,
                             const context &syclContext,
                             const propertyListA &propList = properties{});

template <typename T, typename propertyListA, typename propertyListB>
annotated_ptr<T, propertyListB>
aligned_alloc_host_annotated(size_t alignment, size_t count,
                             const context &syclContext,
                             const propertyListA &propList = properties{}) {
  return {aligned_alloc_host_annotated(alignment, count * sizeof(T),
                                       syclContext, propList)};
}

template <typename propertyListA, typename propertyListB>
annotated_ptr<void, propertyListB>
aligned_alloc_host_annotated(size_t alignment, size_t numBytes,
                             const queue &syclQueue,
                             const propertyListA &propList = properties{}) {
  return aligned_alloc_host_annotated<propertyListA, propertyListB>(
      alignment, numBytes, syclQueue.get_context(), propList);
}

template <typename T, typename propertyListA, typename propertyListB>
annotated_ptr<T, propertyListB>
aligned_alloc_host_annotated(size_t alignment, size_t count,
                             const queue &syclQueue,
                             const propertyListA &propList = properties{}) {
  return aligned_alloc_host_annotated<T, propertyListA, propertyListB>(
      alignment, count, syclQueue.get_context(), propList);
}

////
//  Shared USM allocation functions with properties support
////
template <typename propertyListA, typename propertyListB>
annotated_ptr<void, propertyListB>
malloc_shared_annotated(size_t numBytes, const device &syclDevice,
                        const context &syclContext,
                        const propertyListA &propList = properties{});

template <typename T, typename propertyListA, typename propertyListB>
annotated_ptr<T, propertyListB>
malloc_shared_annotated(size_t count, const device &syclDevice,
                        const context &syclContext,
                        const propertyListA &propList = properties{}) {
  return {aligned_alloc_shared_annotated<propertyListA, propertyListB>(
              align, count * sizeof(T), syclDevice, syclContext, propList)
              .get()};
}

template <typename propertyListA, typename propertyListB>
annotated_ptr<void, propertyListB>
malloc_shared_annotated(size_t numBytes, const queue &syclQueue,
                        const propertyListA &propList = properties{}) {
  return malloc_shared_annotated<propertyListA, propertyListB>(
      numBytes, syclQueue.get_device(), syclQueue.get_context(), propList);
}

template <typename T, typename propertyListA, typename propertyListB>
annotated_ptr<T, propertyListB>
malloc_shared_annotated(size_t count, const queue &syclQueue,
                        const propertyListA &propList = properties{}) {
  return malloc_shared_annotated<T, propertyListA, propertyListB>(
      count, syclQueue.get_device(), syclQueue.get_context(), propList);
}

template <typename propertyListA, typename propertyListB>
annotated_ptr<void, propertyListB> aligned_alloc_shared_annotated(
    size_t alignment, size_t numBytes, const device &syclDevice,
    const context &syclContext, const propertyListA &propList = properties{});

template <typename T, typename propertyListA, typename propertyListB>
annotated_ptr<T, propertyListB> aligned_alloc_shared_annotated(
    size_t alignment, size_t count, const device &syclDevice,
    const context &syclContext, const propertyListA &propList = properties{}) {
  return {aligned_alloc_shared_annotated(alignment, count * sizeof(T),
                                         syclDevice, syclContext, propList)
              .get()};
}

template <typename propertyListA, typename propertyListB>
annotated_ptr<void, propertyListB>
aligned_alloc_shared_annotated(size_t alignment, size_t numBytes,
                               const queue &syclQueue,
                               const propertyListA &propList = properties{}) {
  return aligned_alloc_shared_annotated<propertyListA, propertyListB>(
      alignment, numBytes, syclQueue.get_device(), syclQueue.get_context(),
      propList);
}

template <typename T, typename propertyListA, typename propertyListB>
annotated_ptr<T, propertyListB>
aligned_alloc_shared_annotated(size_t alignment, size_t count,
                               const queue &syclQueue,
                               const propertyListA &propList = properties{}) {
  return aligned_alloc_shared_annotated<T, propertyListA, propertyListB>(
      alignment, count, syclQueue.get_device(), syclQueue.get_context(),
      propList);
}

////
//  Parameterized USM allocation functions with properties support:
//  the usm kind is specified as a function parameter
////
template <typename propertyListA, typename propertyListB>
annotated_ptr<void, propertyListB>
malloc_annotated(size_t numBytes, const device &syclDevice,
                 const context &syclContext, sycl::usm::alloc kind,
                 const propertyListA &propList = properties{});

template <typename T, typename propertyListA, typename propertyListB>
annotated_ptr<T, propertyListB>
malloc_annotated(size_t count, const device &syclDevice,
                 const context &syclContext, sycl::usm::alloc kind,
                 const propertyListA &propList = properties{}) {
  size_t align = get_align_from_property_list(propList);
  return {malloc_annotated<propertyListA, propertyListB>(
              count * sizeof(T), syclDevice, syclContext, kind, propList)
              .get()};
}

template <typename propertyListA, typename propertyListB>
annotated_ptr<void, propertyListB>
malloc_annotated(size_t numBytes, const queue &syclQueue, sycl::usm::alloc kind,
                 const propertyListA &propList = properties{}) {
  return malloc_annotated(numBytes, syclQueue.get_device(),
                          syclQueue.get_context(), kind, propList);
}

template <typename T, typename propertyListA, typename propertyListB>
annotated_ptr<T, propertyListB>
malloc_annotated(size_t count, const queue &syclQueue, sycl::usm::alloc kind,
                 const propertyListA &propList = properties{}) {
  return malloc_annotated(count, syclQueue.get_device(),
                          syclQueue.get_context(), kind, propList);
}

template <typename propertyListA, typename propertyListB>
annotated_ptr<void, propertyListB>
aligned_alloc_annotated(size_t alignment, size_t numBytes,
                        const device &syclDevice, const context &syclContext,
                        sycl::usm::alloc kind,
                        const propertyListA &propList = properties{});

template <typename T, typename propertyListA, typename propertyListB>
annotated_ptr<T, propertyListB>
aligned_alloc_annotated(size_t alignment, size_t count,
                        const device &syclDevice, const context &syclContext,
                        sycl::usm::alloc kind,
                        const propertyListA &propList = properties{}) {
  return {
      aligned_alloc_annotated<propertyListA, propertyListA>(
          alignment, count * sizeof(T), syclDevice, syclContext, kind, propList)
          .get()};
}

template <typename propertyListA, typename propertyListB>
annotated_ptr<void, propertyListB>
aligned_alloc_annotated(size_t alignment, size_t numBytes,
                        const queue &syclQueue, sycl::usm::alloc kind,
                        const propertyListA &propList = properties{}) {
  return aligned_alloc_annotated(alignment, numBytes, syclQueue.get_device(),
                                 syclQueue.get_context(), kind, propList);
}

template <typename T, typename propertyListA, typename propertyListB>
annotated_ptr<T, propertyListB>
aligned_alloc_annotated(size_t alignment, size_t count, const queue &syclQueue,
                        sycl::usm::alloc kind,
                        const propertyListA &propList = properties{}) {
  return aligned_alloc_annotated(alignment, count, syclQueue.get_device(),
                                 syclQueue.get_context(), kind, propList);
}

////
//  Additional USM memory allocation functions:
//  the usm kind of the returned annotated_ptr is specified in the property list
////
template <typename propertyListA, typename propertyListB>
annotated_ptr<void, propertyListB>
malloc_annotated(size_t numBytes, const device &syclDevice,
                 const context &syclContext, const propertyListA &propList);

template <typename T, typename propertyListA, typename propertyListB>
annotated_ptr<T, propertyListB>
malloc_annotated(size_t count, const device &syclDevice,
                 const context &syclContext, const propertyListA &propList) {
  return {malloc_annotated<propertyListA, propertyListB>(
              count * sizeof(T), syclDevice, syclContext, kind, propList)
              .get()};
}

template <typename propertyListA, typename propertyListB>
annotated_ptr<void, propertyListB>
malloc_annotated(size_t numBytes, const queue &syclQueue,
                 const propertyListA &propList) {
  return malloc_annotated<propertyListA, propertyListB>(
      numBytes, syclQueue.get_device(), syclQueue.get_context(), propList);
}

template <typename T, typename propertyListA, typename propertyListB>
annotated_ptr<T, propertyListB>
malloc_annotated(size_t count, const queue &syclQueue,
                 const propertyListA &propList) {
  return malloc_annotated<propertyListA, propertyListB>(
      count, syclQueue.get_device(), syclQueue.get_context(), propList);
}

////
//  Deallocation
////
template <typename T, typename propList>
void free(annotated_ptr<T, propList> &ptr, const context &syclContext) {
  sycl::free(ptr.get(), syclContext);
}

template <typename T, typename propList>
void free(annotated_ptr<T, propList> &ptr, const queue &syclQueue) {
  sycl::free(ptr.get(), syclQueue);
}

} // namespace experimental
} // namespace oneapi
} // namespace ext
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl