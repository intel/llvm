//==----------- buffer.hpp --- SYCL buffer ---------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/common.hpp>
#include <sycl/detail/stl_type_traits.hpp>
#include <sycl/detail/sycl_mem_obj_allocator.hpp>
#include <sycl/event.hpp>
#include <sycl/exception.hpp>
#include <sycl/ext/oneapi/accessor_property_list.hpp>
#include <sycl/property_list.hpp>
#include <sycl/range.hpp>
#include <sycl/stl.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {

class handler;
class queue;
template <int dimensions> class range;

template <typename DataT>
using buffer_allocator = detail::sycl_memory_object_allocator<DataT>;

template <typename DataT, int Dimensions, access::mode AccessMode>
class host_accessor;

template <typename T, int Dimensions, typename AllocatorT, typename Enable>
class buffer;

namespace detail {

class buffer_impl;

template <typename T, int Dimensions, typename AllocatorT>
buffer<T, Dimensions, AllocatorT, void>
make_buffer_helper(pi_native_handle Handle, const context &Ctx, event Evt = {},
                   bool OwnNativeHandle = true) {
  return buffer<T, Dimensions, AllocatorT, void>(Handle, Ctx, OwnNativeHandle,
                                                 Evt);
}

template <backend BackendName, typename DataT, int Dimensions,
          typename Allocator>
auto get_native_buffer(const buffer<DataT, Dimensions, Allocator, void> &Obj)
    -> backend_return_t<BackendName,
                        buffer<DataT, Dimensions, Allocator, void>>;

template <backend Backend, typename DataT, int Dimensions,
          typename AllocatorT = buffer_allocator<std::remove_const_t<DataT>>>
struct BufferInterop;

// The non-template base for the sycl::buffer class
class __SYCL_EXPORT buffer_plain {
protected:
  buffer_plain(size_t SizeInBytes, size_t, const property_list &Props,
               std::unique_ptr<detail::SYCLMemObjAllocator> Allocator);

  buffer_plain(void *HostData, size_t SizeInBytes, size_t RequiredAlign,
               const property_list &Props,
               std::unique_ptr<detail::SYCLMemObjAllocator> Allocator);

  buffer_plain(const void *HostData, size_t SizeInBytes, size_t RequiredAlign,
               const property_list &Props,
               std::unique_ptr<detail::SYCLMemObjAllocator> Allocator);

  buffer_plain(const std::shared_ptr<const void> &HostData,
               const size_t SizeInBytes, size_t RequiredAlign,
               const property_list &Props,
               std::unique_ptr<detail::SYCLMemObjAllocator> Allocator,
               bool IsConstPtr);

  buffer_plain(const std::function<void(void *)>
                   &CopyFromInput, // EnableIfNotConstIterator<InputIterator>
                                   // First, InputIterator Last,
               const size_t SizeInBytes, size_t RequiredAlign,
               const property_list &Props,
               std::unique_ptr<detail::SYCLMemObjAllocator> Allocator,
               bool IsConstPtr);

  buffer_plain(pi_native_handle MemObject, context SyclContext,
               std::unique_ptr<detail::SYCLMemObjAllocator> Allocator,
               bool OwnNativeHandle, event AvailableEvent);

  buffer_plain(const std::shared_ptr<detail::buffer_impl> &impl) : impl(impl) {}

  void set_final_data_internal();

  void set_final_data_internal(
      const std::function<void(const std::function<void(void *const Ptr)> &)>
          &FinalDataFunc);

  void set_write_back(bool NeedWriteBack);

  void constructorNotification(const detail::code_location &CodeLoc,
                               void *UserObj, const void *HostObj,
                               const void *Type, uint32_t Dim,
                               uint32_t ElemType, size_t Range[3]);

  template <typename propertyT> bool has_property() const noexcept;

  template <typename propertyT> propertyT get_property() const;

  std::vector<pi_native_handle> getNativeVector(backend BackendName) const;

  const std::unique_ptr<SYCLMemObjAllocator> &get_allocator_internal() const;

  void deleteAccProps(const sycl::detail::PropWithDataKind &Kind);

  void addOrReplaceAccessorProperties(const property_list &PropertyList);

  size_t getSize() const;

  std::shared_ptr<detail::buffer_impl> impl;
};

} // namespace detail

/// Defines a shared array that can be used by kernels in queues.
///
/// Buffers can be 1-, 2-, and 3-dimensional. They have to be accessed using
/// accessor classes.
///
/// \sa sycl_api_acc
///
/// \ingroup sycl_api
template <typename T, int dimensions = 1,
          typename AllocatorT = buffer_allocator<std::remove_const_t<T>>,
          typename __Enabled = typename detail::enable_if_t<(dimensions > 0) &&
                                                            (dimensions <= 3)>>
class buffer : public detail::buffer_plain {
  // TODO check is_device_copyable<T>::value after converting sycl::vec into a
  // trivially copyable class.
  static_assert(!std::is_same<T, std::string>::value,
                "'std::string' is not a device copyable type");

public:
  using value_type = T;
  using reference = value_type &;
  using const_reference = const value_type &;
  using allocator_type = AllocatorT;
  template <int dims>
  using EnableIfOneDimension = typename detail::enable_if_t<1 == dims>;
  // using same requirement for contiguous container as std::span
  template <class Container>
  using EnableIfContiguous =
      detail::void_t<detail::enable_if_t<std::is_convertible<
                         detail::remove_pointer_t<
                             decltype(std::declval<Container>().data())> (*)[],
                         const T (*)[]>::value>,
                     decltype(std::declval<Container>().size())>;
  template <class It>
  using EnableIfItInputIterator = detail::enable_if_t<
      std::is_convertible<typename std::iterator_traits<It>::iterator_category,
                          std::input_iterator_tag>::value>;
  template <typename ItA, typename ItB>
  using EnableIfSameNonConstIterators = typename detail::enable_if_t<
      std::is_same<ItA, ItB>::value && !std::is_const<ItA>::value, ItA>;

  std::array<size_t, 3> rangeToArray(range<3> &r) { return {r[0], r[1], r[2]}; }

  std::array<size_t, 3> rangeToArray(range<2> &r) { return {r[0], r[1], 0}; }

  std::array<size_t, 3> rangeToArray(range<1> &r) { return {r[0], 0, 0}; }

  buffer(const range<dimensions> &bufferRange,
         const property_list &propList = {},
         const detail::code_location CodeLoc = detail::code_location::current())
      : buffer_plain(bufferRange.size() * sizeof(T),
                     detail::getNextPowerOfTwo(sizeof(T)), propList,
                     make_unique_ptr<
                         detail::SYCLMemObjAllocatorHolder<AllocatorT, T>>()),
        Range(bufferRange) {
    buffer_plain::constructorNotification(
        CodeLoc, (void *)impl.get(), nullptr, (const void *)typeid(T).name(),
        dimensions, sizeof(T), rangeToArray(Range).data());
  }

  buffer(const range<dimensions> &bufferRange, AllocatorT allocator,
         const property_list &propList = {},
         const detail::code_location CodeLoc = detail::code_location::current())
      : buffer_plain(
            bufferRange.size() * sizeof(T),
            detail::getNextPowerOfTwo(sizeof(T)), propList,
            make_unique_ptr<detail::SYCLMemObjAllocatorHolder<AllocatorT, T>>(
                allocator)),
        Range(bufferRange) {
    buffer_plain::constructorNotification(
        CodeLoc, (void *)impl.get(), nullptr, (const void *)typeid(T).name(),
        dimensions, sizeof(T), rangeToArray(Range).data());
  }

  buffer(T *hostData, const range<dimensions> &bufferRange,
         const property_list &propList = {},
         const detail::code_location CodeLoc = detail::code_location::current())
      : buffer_plain(hostData, bufferRange.size() * sizeof(T),
                     detail::getNextPowerOfTwo(sizeof(T)), propList,
                     make_unique_ptr<
                         detail::SYCLMemObjAllocatorHolder<AllocatorT, T>>()),
        Range(bufferRange) {
    buffer_plain::constructorNotification(
        CodeLoc, (void *)impl.get(), hostData, (const void *)typeid(T).name(),
        dimensions, sizeof(T), rangeToArray(Range).data());
  }

  buffer(T *hostData, const range<dimensions> &bufferRange,
         AllocatorT allocator, const property_list &propList = {},
         const detail::code_location CodeLoc = detail::code_location::current())
      : buffer_plain(
            hostData, bufferRange.size() * sizeof(T),
            detail::getNextPowerOfTwo(sizeof(T)), propList,
            make_unique_ptr<detail::SYCLMemObjAllocatorHolder<AllocatorT, T>>(
                allocator)),
        Range(bufferRange) {
    buffer_plain::constructorNotification(
        CodeLoc, (void *)impl.get(), hostData, (const void *)typeid(T).name(),
        dimensions, sizeof(T), rangeToArray(Range).data());
  }

  template <typename _T = T>
  buffer(EnableIfSameNonConstIterators<T, _T> const *hostData,
         const range<dimensions> &bufferRange,
         const property_list &propList = {},
         const detail::code_location CodeLoc = detail::code_location::current())
      : buffer_plain(hostData, bufferRange.size() * sizeof(T),
                     detail::getNextPowerOfTwo(sizeof(T)), propList,
                     make_unique_ptr<
                         detail::SYCLMemObjAllocatorHolder<AllocatorT, T>>()),
        Range(bufferRange) {
    buffer_plain::constructorNotification(
        CodeLoc, (void *)impl.get(), hostData, (const void *)typeid(T).name(),
        dimensions, sizeof(T), rangeToArray(Range).data());
  }

  template <typename _T = T>
  buffer(EnableIfSameNonConstIterators<T, _T> const *hostData,
         const range<dimensions> &bufferRange, AllocatorT allocator,
         const property_list &propList = {},
         const detail::code_location CodeLoc = detail::code_location::current())
      : buffer_plain(
            bufferRange.size() * sizeof(T),
            detail::getNextPowerOfTwo(sizeof(T)), propList,
            make_unique_ptr<detail::SYCLMemObjAllocatorHolder<AllocatorT, T>>(
                allocator)),
        Range(bufferRange) {
    buffer_plain::constructorNotification(
        CodeLoc, (void *)impl.get(), hostData, (const void *)typeid(T).name(),
        dimensions, sizeof(T), rangeToArray(Range).data());
  }

  buffer(const std::shared_ptr<T> &hostData,
         const range<dimensions> &bufferRange, AllocatorT allocator,
         const property_list &propList = {},
         const detail::code_location CodeLoc = detail::code_location::current())
      : buffer_plain(
            hostData, bufferRange.size() * sizeof(T),
            detail::getNextPowerOfTwo(sizeof(T)), propList,
            make_unique_ptr<detail::SYCLMemObjAllocatorHolder<AllocatorT, T>>(
                allocator),
            std::is_const<T>::value),
        Range(bufferRange) {
    buffer_plain::constructorNotification(
        CodeLoc, (void *)impl.get(), (void *)hostData.get(),
        (const void *)typeid(T).name(), dimensions, sizeof(T),
        rangeToArray(Range).data());
  }

  buffer(const std::shared_ptr<T[]> &hostData,
         const range<dimensions> &bufferRange, AllocatorT allocator,
         const property_list &propList = {},
         const detail::code_location CodeLoc = detail::code_location::current())
      : buffer_plain(
            hostData, bufferRange.size() * sizeof(T),
            detail::getNextPowerOfTwo(sizeof(T)), propList,
            make_unique_ptr<detail::SYCLMemObjAllocatorHolder<AllocatorT, T>>(
                allocator),
            std::is_const<T>::value),
        Range(bufferRange) {
    buffer_plain::constructorNotification(
        CodeLoc, (void *)impl.get(), (void *)hostData.get(),
        (const void *)typeid(T).name(), dimensions, sizeof(T),
        rangeToArray(Range).data());
  }

  buffer(const std::shared_ptr<T> &hostData,
         const range<dimensions> &bufferRange,
         const property_list &propList = {},
         const detail::code_location CodeLoc = detail::code_location::current())
      : buffer_plain(
            hostData, bufferRange.size() * sizeof(T),
            detail::getNextPowerOfTwo(sizeof(T)), propList,
            make_unique_ptr<detail::SYCLMemObjAllocatorHolder<AllocatorT, T>>(),
            std::is_const<T>::value),
        Range(bufferRange) {
    buffer_plain::constructorNotification(
        CodeLoc, (void *)impl.get(), (void *)hostData.get(),
        (const void *)typeid(T).name(), dimensions, sizeof(T),
        rangeToArray(Range).data());
  }

  buffer(const std::shared_ptr<T[]> &hostData,
         const range<dimensions> &bufferRange,
         const property_list &propList = {},
         const detail::code_location CodeLoc = detail::code_location::current())
      : buffer_plain(
            hostData, bufferRange.size() * sizeof(T),
            detail::getNextPowerOfTwo(sizeof(T)), propList,
            make_unique_ptr<detail::SYCLMemObjAllocatorHolder<AllocatorT, T>>(),
            std::is_const<T>::value),
        Range(bufferRange) {
    buffer_plain::constructorNotification(
        CodeLoc, (void *)impl.get(), (void *)hostData.get(),
        (const void *)typeid(T).name(), dimensions, sizeof(T),
        rangeToArray(Range).data());
  }

  template <class InputIterator, int N = dimensions,
            typename = EnableIfOneDimension<N>,
            typename = EnableIfItInputIterator<InputIterator>>
  buffer(InputIterator first, InputIterator last, AllocatorT allocator,
         const property_list &propList = {},
         const detail::code_location CodeLoc = detail::code_location::current())
      : buffer_plain(
            // The functor which will be used to initialize the data
            [first, last](void *ToPtr) {
              // We need to cast MUserPtr to pointer to the iteration type to
              // get correct offset in std::copy when it will increment
              // destination pointer.
              using IteratorValueType =
                  detail::iterator_value_type_t<InputIterator>;
              using IteratorNonConstValueType =
                  detail::remove_const_t<IteratorValueType>;
              using IteratorPointerToNonConstValueType =
                  detail::add_pointer_t<IteratorNonConstValueType>;
              std::copy(first, last,
                        static_cast<IteratorPointerToNonConstValueType>(ToPtr));
            },
            std::distance(first, last) * sizeof(T),
            detail::getNextPowerOfTwo(sizeof(T)), propList,
            make_unique_ptr<detail::SYCLMemObjAllocatorHolder<AllocatorT, T>>(
                allocator),
            detail::iterator_to_const_type_t<InputIterator>::value),
        Range(range<1>(std::distance(first, last))) {
    size_t r[3] = {Range[0], 0, 0};
    buffer_plain::constructorNotification(CodeLoc, (void *)impl.get(), &first,
                                          (const void *)typeid(T).name(),
                                          dimensions, sizeof(T), r);
  }

  template <class InputIterator, int N = dimensions,
            typename = EnableIfOneDimension<N>,
            typename = EnableIfItInputIterator<InputIterator>>
  buffer(InputIterator first, InputIterator last,
         const property_list &propList = {},
         const detail::code_location CodeLoc = detail::code_location::current())
      : buffer_plain(
            // The functor which will be used to initialize the data
            [first, last](void *ToPtr) {
              // We need to cast MUserPtr to pointer to the iteration type to
              // get correct offset in std::copy when it will increment
              // destination pointer.
              using IteratorValueType =
                  detail::iterator_value_type_t<InputIterator>;
              using IteratorNonConstValueType =
                  detail::remove_const_t<IteratorValueType>;
              using IteratorPointerToNonConstValueType =
                  detail::add_pointer_t<IteratorNonConstValueType>;
              std::copy(first, last,
                        static_cast<IteratorPointerToNonConstValueType>(ToPtr));
            },
            std::distance(first, last) * sizeof(T),
            detail::getNextPowerOfTwo(sizeof(T)), propList,
            make_unique_ptr<detail::SYCLMemObjAllocatorHolder<AllocatorT, T>>(),
            detail::iterator_to_const_type_t<InputIterator>::value),
        Range(range<1>(std::distance(first, last))) {
    size_t r[3] = {Range[0], 0, 0};
    buffer_plain::constructorNotification(CodeLoc, (void *)impl.get(), &first,
                                          (const void *)typeid(T).name(),
                                          dimensions, sizeof(T), r);
  }

  // This constructor is a prototype for a future SYCL specification
  template <class Container, int N = dimensions,
            typename = EnableIfOneDimension<N>,
            typename = EnableIfContiguous<Container>>
  buffer(Container &container, AllocatorT allocator,
         const property_list &propList = {},
         const detail::code_location CodeLoc = detail::code_location::current())
      : buffer_plain(
            container.data(), container.size() * sizeof(T),
            detail::getNextPowerOfTwo(sizeof(T)), propList,
            make_unique_ptr<detail::SYCLMemObjAllocatorHolder<AllocatorT, T>>(
                allocator)),
        Range(range<1>(container.size())) {
    size_t r[3] = {Range[0], 0, 0};
    buffer_plain::constructorNotification(
        CodeLoc, (void *)impl.get(), container.data(),
        (const void *)typeid(T).name(), dimensions, sizeof(T), r);
  }

  // This constructor is a prototype for a future SYCL specification
  template <class Container, int N = dimensions,
            typename = EnableIfOneDimension<N>,
            typename = EnableIfContiguous<Container>>
  buffer(Container &container, const property_list &propList = {},
         const detail::code_location CodeLoc = detail::code_location::current())
      : buffer(container, {}, propList, CodeLoc) {}

  buffer(buffer<T, dimensions, AllocatorT> &b, const id<dimensions> &baseIndex,
         const range<dimensions> &subRange,
         const detail::code_location CodeLoc = detail::code_location::current())
      : buffer_plain(b.impl), Range(subRange),
        OffsetInBytes(getOffsetInBytes<T>(baseIndex, b.Range)),
        IsSubBuffer(true) {
    buffer_plain::constructorNotification(
        CodeLoc, (void *)impl.get(), impl.get(), (const void *)typeid(T).name(),
        dimensions, sizeof(T), rangeToArray(Range).data());

    if (b.is_sub_buffer())
      throw sycl::invalid_object_error(
          "Cannot create sub buffer from sub buffer.", PI_ERROR_INVALID_VALUE);
    if (isOutOfBounds(baseIndex, subRange, b.Range))
      throw sycl::invalid_object_error(
          "Requested sub-buffer size exceeds the size of the parent buffer",
          PI_ERROR_INVALID_VALUE);
    if (!isContiguousRegion(baseIndex, subRange, b.Range))
      throw sycl::invalid_object_error(
          "Requested sub-buffer region is not contiguous",
          PI_ERROR_INVALID_VALUE);
  }

  buffer(const buffer &rhs,
         const detail::code_location CodeLoc = detail::code_location::current())
      : buffer_plain(rhs.impl), Range(rhs.Range),
        OffsetInBytes(rhs.OffsetInBytes), IsSubBuffer(rhs.IsSubBuffer) {
    buffer_plain::constructorNotification(
        CodeLoc, (void *)impl.get(), impl.get(), (const void *)typeid(T).name(),
        dimensions, sizeof(T), rangeToArray(Range).data());
  }

  buffer(buffer &&rhs,
         const detail::code_location CodeLoc = detail::code_location::current())
      : buffer_plain(std::move(rhs.impl)), Range(rhs.Range),
        OffsetInBytes(rhs.OffsetInBytes), IsSubBuffer(rhs.IsSubBuffer) {
    buffer_plain::constructorNotification(
        CodeLoc, (void *)impl.get(), impl.get(), (const void *)typeid(T).name(),
        dimensions, sizeof(T), rangeToArray(Range).data());
  }

  buffer &operator=(const buffer &rhs) = default;

  buffer &operator=(buffer &&rhs) = default;

  ~buffer() = default;

  bool operator==(const buffer &rhs) const { return impl == rhs.impl; }

  bool operator!=(const buffer &rhs) const { return !(*this == rhs); }

  /* -- common interface members -- */

  /* -- property interface members -- */

  range<dimensions> get_range() const { return Range; }

  __SYCL2020_DEPRECATED("get_count() is deprecated, please use size() instead")
  size_t get_count() const { return size(); }
  size_t size() const noexcept { return Range.size(); }

  __SYCL2020_DEPRECATED(
      "get_size() is deprecated, please use byte_size() instead")
  size_t get_size() const { return byte_size(); }
  size_t byte_size() const noexcept { return size() * sizeof(T); }

  AllocatorT get_allocator() const {
    return buffer_plain::get_allocator_internal()
        ->template getAllocator<AllocatorT>();
  }

  template <access::mode Mode, access::target Target = access::target::device>
  accessor<T, dimensions, Mode, Target, access::placeholder::false_t,
           ext::oneapi::accessor_property_list<>>
  get_access(
      handler &CommandGroupHandler,
      const detail::code_location CodeLoc = detail::code_location::current()) {
    return accessor<T, dimensions, Mode, Target, access::placeholder::false_t,
                    ext::oneapi::accessor_property_list<>>(
        *this, CommandGroupHandler, {}, CodeLoc);
  }

  template <access::mode mode>
  accessor<T, dimensions, mode, access::target::host_buffer,
           access::placeholder::false_t, ext::oneapi::accessor_property_list<>>
  get_access(
      const detail::code_location CodeLoc = detail::code_location::current()) {
    return accessor<T, dimensions, mode, access::target::host_buffer,
                    access::placeholder::false_t,
                    ext::oneapi::accessor_property_list<>>(*this, {}, CodeLoc);
  }

  template <access::mode mode, access::target target = access::target::device>
  accessor<T, dimensions, mode, target, access::placeholder::false_t,
           ext::oneapi::accessor_property_list<>>
  get_access(
      handler &commandGroupHandler, range<dimensions> accessRange,
      id<dimensions> accessOffset = {},
      const detail::code_location CodeLoc = detail::code_location::current()) {
    if (isOutOfBounds(accessOffset, accessRange, this->Range))
      throw sycl::invalid_object_error(
          "Requested accessor would exceed the bounds of the buffer",
          PI_ERROR_INVALID_VALUE);

    return accessor<T, dimensions, mode, target, access::placeholder::false_t,
                    ext::oneapi::accessor_property_list<>>(
        *this, commandGroupHandler, accessRange, accessOffset, {}, CodeLoc);
  }

  template <access::mode mode>
  accessor<T, dimensions, mode, access::target::host_buffer,
           access::placeholder::false_t, ext::oneapi::accessor_property_list<>>
  get_access(
      range<dimensions> accessRange, id<dimensions> accessOffset = {},
      const detail::code_location CodeLoc = detail::code_location::current()) {
    if (isOutOfBounds(accessOffset, accessRange, this->Range))
      throw sycl::invalid_object_error(
          "Requested accessor would exceed the bounds of the buffer",
          PI_ERROR_INVALID_VALUE);

    return accessor<T, dimensions, mode, access::target::host_buffer,
                    access::placeholder::false_t,
                    ext::oneapi::accessor_property_list<>>(
        *this, accessRange, accessOffset, {}, CodeLoc);
  }

#if __cplusplus >= 201703L

  template <typename... Ts> auto get_access(Ts... args) {
    return accessor{*this, args...};
  }

  template <typename... Ts>
  auto get_access(handler &commandGroupHandler, Ts... args) {
    return accessor{*this, commandGroupHandler, args...};
  }

  template <typename... Ts> auto get_host_access(Ts... args) {
    return host_accessor{*this, args...};
  }

  template <typename... Ts>
  auto get_host_access(handler &commandGroupHandler, Ts... args) {
    return host_accessor{*this, commandGroupHandler, args...};
  }

#endif

  template <typename Destination = std::nullptr_t>
  void set_final_data(Destination finalData = nullptr) {
    this->set_final_data_internal(finalData);
  }

  void set_final_data_internal(std::nullptr_t) {
    buffer_plain::set_final_data_internal();
  }

  template <template <typename WeakT> class WeakPtrT, typename WeakT>
  detail::enable_if_t<
      std::is_convertible<WeakPtrT<WeakT>, std::weak_ptr<WeakT>>::value>
  set_final_data_internal(WeakPtrT<WeakT> FinalData) {
    std::weak_ptr<WeakT> TempFinalData(FinalData);
    this->set_final_data_internal(TempFinalData);
  }

  template <typename WeakT>
  void set_final_data_internal(std::weak_ptr<WeakT> FinalData) {
    buffer_plain::set_final_data_internal(
        [FinalData](const std::function<void(void *const Ptr)> &F) {
          if (std::shared_ptr<WeakT> LockedFinalData = FinalData.lock())
            F(LockedFinalData.get());
        });
  }

  template <typename Destination>
  detail::EnableIfOutputPointerT<Destination>
  set_final_data_internal(Destination FinalData) {
    if (!FinalData)
      buffer_plain::set_final_data_internal();
    else
      buffer_plain::set_final_data_internal(
          [FinalData](const std::function<void(void *const Ptr)> &F) {
            F(FinalData);
          });
  }

  template <typename Destination>
  detail::EnableIfOutputIteratorT<Destination>
  set_final_data_internal(Destination FinalData) {
    const size_t Size = size();
    buffer_plain::set_final_data_internal(
        [FinalData, Size](const std::function<void(void *const Ptr)> &F) {
          using DestinationValueT = detail::iterator_value_type_t<Destination>;
          // TODO if Destination is ContiguousIterator then don't create
          // ContiguousStorage. updateHostMemory works only with pointer to
          // continuous data.
          std::unique_ptr<DestinationValueT[]> ContiguousStorage(
              new DestinationValueT[Size]);
          F(ContiguousStorage.get());
          std::copy(ContiguousStorage.get(), ContiguousStorage.get() + Size,
                    FinalData);
        });
  }

  void set_final_data(std::nullptr_t) {
    buffer_plain::set_final_data_internal();
  }

  void set_write_back(bool flag = true) { buffer_plain::set_write_back(flag); }

  bool is_sub_buffer() const { return IsSubBuffer; }

  template <typename ReinterpretT, int ReinterpretDim>
  buffer<ReinterpretT, ReinterpretDim,
         typename std::allocator_traits<AllocatorT>::template rebind_alloc<
             std::remove_const_t<ReinterpretT>>>
  reinterpret(range<ReinterpretDim> reinterpretRange) const {
    if (sizeof(ReinterpretT) * reinterpretRange.size() != byte_size())
      throw sycl::invalid_object_error(
          "Total size in bytes represented by the type and range of the "
          "reinterpreted SYCL buffer does not equal the total size in bytes "
          "represented by the type and range of this SYCL buffer",
          PI_ERROR_INVALID_VALUE);

    return buffer<ReinterpretT, ReinterpretDim,
                  typename std::allocator_traits<AllocatorT>::
                      template rebind_alloc<std::remove_const_t<ReinterpretT>>>(
        impl, reinterpretRange, OffsetInBytes, IsSubBuffer);
  }

  template <typename ReinterpretT, int ReinterpretDim = dimensions>
  typename std::enable_if<
      (sizeof(ReinterpretT) == sizeof(T)) && (dimensions == ReinterpretDim),
      buffer<ReinterpretT, ReinterpretDim,
             typename std::allocator_traits<AllocatorT>::template rebind_alloc<
                 std::remove_const_t<ReinterpretT>>>>::type
  reinterpret() const {
    return buffer<ReinterpretT, ReinterpretDim,
                  typename std::allocator_traits<AllocatorT>::
                      template rebind_alloc<std::remove_const_t<ReinterpretT>>>(
        impl, get_range(), OffsetInBytes, IsSubBuffer);
  }

  template <typename ReinterpretT, int ReinterpretDim = dimensions>
  typename std::enable_if<
      (ReinterpretDim == 1) && ((dimensions != ReinterpretDim) ||
                                (sizeof(ReinterpretT) != sizeof(T))),
      buffer<ReinterpretT, ReinterpretDim, AllocatorT>>::type
  reinterpret() const {
    long sz = byte_size();
    if (sz % sizeof(ReinterpretT) != 0)
      throw sycl::invalid_object_error(
          "Total byte size of buffer is not evenly divisible by the size of "
          "the reinterpreted type",
          PI_ERROR_INVALID_VALUE);

    return buffer<ReinterpretT, ReinterpretDim, AllocatorT>(
        impl, range<1>{sz / sizeof(ReinterpretT)}, OffsetInBytes, IsSubBuffer);
  }

  template <typename propertyT> bool has_property() const noexcept {
    return buffer_plain::template has_property<propertyT>();
  }

  template <typename propertyT> propertyT get_property() const {
    return buffer_plain::template get_property<propertyT>();
  }

protected:
  bool isOutOfBounds(const id<dimensions> &offset,
                     const range<dimensions> &newRange,
                     const range<dimensions> &parentRange) {
    bool outOfBounds = false;
    for (int i = 0; i < dimensions; ++i)
      outOfBounds |= newRange[i] + offset[i] > parentRange[i];

    return outOfBounds;
  }

private:
  template <class Obj>
  friend decltype(Obj::impl) detail::getSyclObjImpl(const Obj &SyclObject);
  template <typename A, int dims, typename C, typename Enable>
  friend class buffer;
  template <typename DataT, int dims, access::mode mode, access::target target,
            access::placeholder isPlaceholder, typename PropertyListT>
  friend class accessor;
  template <typename HT, int HDims, typename HAllocT>
  friend buffer<HT, HDims, HAllocT, void>
  detail::make_buffer_helper(pi_native_handle, const context &, event, bool);
  range<dimensions> Range;
  // Offset field specifies the origin of the sub buffer inside the parent
  // buffer
  size_t OffsetInBytes = 0;
  bool IsSubBuffer = false;

  // Interop constructor
  template <int N = dimensions, typename = EnableIfOneDimension<N>>
  buffer(pi_native_handle MemObject, const context &SyclContext,
         bool OwnNativeHandle, event AvailableEvent = {},
         const detail::code_location CodeLoc = detail::code_location::current())
      : buffer_plain(
            MemObject, SyclContext,
            make_unique_ptr<detail::SYCLMemObjAllocatorHolder<AllocatorT, T>>(),
            OwnNativeHandle, std::move(AvailableEvent)),
        Range{0} {

    Range[0] = buffer_plain::getSize() / sizeof(T);
    buffer_plain::constructorNotification(
        CodeLoc, (void *)impl.get(), &MemObject, (const void *)typeid(T).name(),
        dimensions, sizeof(T), rangeToArray(Range).data());
  }

  void addOrReplaceAccessorProperties(const property_list &PropertyList) {
    buffer_plain::addOrReplaceAccessorProperties(PropertyList);
  }

  void deleteAccProps(const sycl::detail::PropWithDataKind &Kind) {
    buffer_plain::deleteAccProps(Kind);
  }

  // Reinterpret contructor
  buffer(const std::shared_ptr<detail::buffer_impl> &Impl,
         range<dimensions> reinterpretRange, size_t reinterpretOffset,
         bool isSubBuffer,
         const detail::code_location CodeLoc = detail::code_location::current())
      : buffer_plain(Impl), Range(reinterpretRange),
        OffsetInBytes(reinterpretOffset), IsSubBuffer(isSubBuffer) {
    buffer_plain::constructorNotification(
        CodeLoc, (void *)impl.get(), Impl.get(), (const void *)typeid(T).name(),
        dimensions, sizeof(T), rangeToArray(Range).data());
  }

  template <typename Type, int N>
  size_t getOffsetInBytes(const id<N> &offset, const range<N> &range) {
    return detail::getLinearIndex(offset, range) * sizeof(Type);
  }

  bool isContiguousRegion(const id<1> &, const range<1> &, const range<1> &) {
    // 1D sub buffer always has contiguous region
    return true;
  }

  bool isContiguousRegion(const id<2> &offset, const range<2> &newRange,
                          const range<2> &parentRange) {
    // For 2D sub buffer there are 2 cases:
    // 1) Offset {Any, Any}  | a piece of any line of a buffer
    //    Range  {1,   Any}  |
    // 2) Offset {Any, 0  }  | any number of full lines
    //    Range  {Any, Col}  |
    // where Col is a number of columns of original buffer
    if (offset[1])
      return newRange[0] == 1;
    return newRange[1] == parentRange[1];
  }

  bool isContiguousRegion(const id<3> &offset, const range<3> &newRange,
                          const range<3> &parentRange) {
    // For 3D sub buffer there are 3 cases:
    // 1) Offset {Any, Any, Any}  | a piece of any line in any slice of a buffer
    //    Range  {1,   1,   Any}  |
    // 2) Offset {Any, Any, 0  }  | any number of full lines in any slice
    //    Range  {1,   Any, Col}  |
    // 3) Offset {Any, 0,   0  }  | any number of slices
    //    Range  {Any, Row, Col}  |
    // where Row and Col are numbers of rows and columns of original buffer
    if (offset[2])
      return newRange[0] == 1 && newRange[1] == 1;
    if (offset[1])
      return newRange[0] == 1 && newRange[2] == parentRange[2];
    return newRange[1] == parentRange[1] && newRange[2] == parentRange[2];
  }

  template <backend BackendName, typename DataT, int Dimensions,
            typename Allocator>
  friend auto detail::get_native_buffer(
      const buffer<DataT, Dimensions, Allocator, void> &Obj)
      -> backend_return_t<BackendName,
                          buffer<DataT, Dimensions, Allocator, void>>;

  template <backend BackendName>
  backend_return_t<BackendName, buffer<T, dimensions, AllocatorT>>
  getNative() const {
    auto NativeHandles = buffer_plain::getNativeVector(BackendName);
    return detail::BufferInterop<BackendName, T, dimensions,
                                 AllocatorT>::GetNativeObjs(NativeHandles);
  }
};

#ifdef __cpp_deduction_guides
template <class InputIterator, class AllocatorT>
buffer(InputIterator, InputIterator, AllocatorT, const property_list & = {})
    -> buffer<typename std::iterator_traits<InputIterator>::value_type, 1,
              AllocatorT>;
template <class InputIterator>
buffer(InputIterator, InputIterator, const property_list & = {})
    -> buffer<typename std::iterator_traits<InputIterator>::value_type, 1>;
template <class Container, class AllocatorT>
buffer(Container &, AllocatorT, const property_list & = {})
    -> buffer<typename Container::value_type, 1, AllocatorT>;
template <class Container>
buffer(Container &, const property_list & = {})
    -> buffer<typename Container::value_type, 1>;
template <class T, int dimensions, class AllocatorT>
buffer(const T *, const range<dimensions> &, AllocatorT,
       const property_list & = {}) -> buffer<T, dimensions, AllocatorT>;
template <class T, int dimensions>
buffer(const T *, const range<dimensions> &, const property_list & = {})
    -> buffer<T, dimensions>;
#endif // __cpp_deduction_guides

} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl

namespace std {
template <typename T, int dimensions, typename AllocatorT>
struct hash<sycl::buffer<T, dimensions, AllocatorT>> {
  size_t operator()(const sycl::buffer<T, dimensions, AllocatorT> &b) const {
    return hash<std::shared_ptr<sycl::detail::buffer_impl>>()(
        sycl::detail::getSyclObjImpl(b));
  }
};
} // namespace std
