//==----------- buffer.hpp --- SYCL buffer ---------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/buffer_impl.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/detail/stl_type_traits.hpp>
#include <sycl/exception.hpp>
#include <sycl/ext/oneapi/accessor_property_list.hpp>
#include <sycl/property_list.hpp>
#include <sycl/stl.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {

class handler;
class queue;
template <int dimensions> class range;

template <typename DataT>
using buffer_allocator = detail::sycl_memory_object_allocator<DataT>;

namespace detail {

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
class buffer {
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
      : Range(bufferRange) {
    impl = std::make_shared<detail::buffer_impl>(
        size() * sizeof(T), detail::getNextPowerOfTwo(sizeof(T)), propList,
        make_unique_ptr<detail::SYCLMemObjAllocatorHolder<AllocatorT, T>>());
    impl->constructorNotification(CodeLoc, (void *)impl.get(), nullptr,
                                  (const void *)typeid(T).name(), dimensions,
                                  sizeof(T), rangeToArray(Range).data());
  }

  buffer(const range<dimensions> &bufferRange, AllocatorT allocator,
         const property_list &propList = {},
         const detail::code_location CodeLoc = detail::code_location::current())
      : Range(bufferRange) {
    impl = std::make_shared<detail::buffer_impl>(
        size() * sizeof(T), detail::getNextPowerOfTwo(sizeof(T)), propList,
        make_unique_ptr<detail::SYCLMemObjAllocatorHolder<AllocatorT, T>>(
            allocator));
    impl->constructorNotification(CodeLoc, (void *)impl.get(), nullptr,
                                  (const void *)typeid(T).name(), dimensions,
                                  sizeof(T), rangeToArray(Range).data());
  }

  buffer(T *hostData, const range<dimensions> &bufferRange,
         const property_list &propList = {},
         const detail::code_location CodeLoc = detail::code_location::current())
      : Range(bufferRange) {
    impl = std::make_shared<detail::buffer_impl>(
        hostData, size() * sizeof(T), detail::getNextPowerOfTwo(sizeof(T)),
        propList,
        make_unique_ptr<detail::SYCLMemObjAllocatorHolder<AllocatorT, T>>());
    impl->constructorNotification(CodeLoc, (void *)impl.get(), hostData,
                                  (const void *)typeid(T).name(), dimensions,
                                  sizeof(T), rangeToArray(Range).data());
  }

  buffer(T *hostData, const range<dimensions> &bufferRange,
         AllocatorT allocator, const property_list &propList = {},
         const detail::code_location CodeLoc = detail::code_location::current())
      : Range(bufferRange) {
    impl = std::make_shared<detail::buffer_impl>(
        hostData, size() * sizeof(T), detail::getNextPowerOfTwo(sizeof(T)),
        propList,
        make_unique_ptr<detail::SYCLMemObjAllocatorHolder<AllocatorT, T>>(
            allocator));
    impl->constructorNotification(CodeLoc, (void *)impl.get(), hostData,
                                  (const void *)typeid(T).name(), dimensions,
                                  sizeof(T), rangeToArray(Range).data());
  }

  template <typename _T = T>
  buffer(EnableIfSameNonConstIterators<T, _T> const *hostData,
         const range<dimensions> &bufferRange,
         const property_list &propList = {},
         const detail::code_location CodeLoc = detail::code_location::current())
      : Range(bufferRange) {
    impl = std::make_shared<detail::buffer_impl>(
        hostData, size() * sizeof(T), detail::getNextPowerOfTwo(sizeof(T)),
        propList,
        make_unique_ptr<detail::SYCLMemObjAllocatorHolder<AllocatorT, T>>());
    impl->constructorNotification(CodeLoc, (void *)impl.get(), hostData,
                                  (const void *)typeid(T).name(), dimensions,
                                  sizeof(T), rangeToArray(Range).data());
  }

  template <typename _T = T>
  buffer(EnableIfSameNonConstIterators<T, _T> const *hostData,
         const range<dimensions> &bufferRange, AllocatorT allocator,
         const property_list &propList = {},
         const detail::code_location CodeLoc = detail::code_location::current())
      : Range(bufferRange) {
    impl = std::make_shared<detail::buffer_impl>(
        hostData, size() * sizeof(T), detail::getNextPowerOfTwo(sizeof(T)),
        propList,
        make_unique_ptr<detail::SYCLMemObjAllocatorHolder<AllocatorT, T>>(
            allocator));
    impl->constructorNotification(CodeLoc, (void *)impl.get(), hostData,
                                  (const void *)typeid(T).name(), dimensions,
                                  sizeof(T), rangeToArray(Range).data());
  }

  buffer(const std::shared_ptr<T> &hostData,
         const range<dimensions> &bufferRange, AllocatorT allocator,
         const property_list &propList = {},
         const detail::code_location CodeLoc = detail::code_location::current())
      : Range(bufferRange) {
    impl = std::make_shared<detail::buffer_impl>(
        hostData, size() * sizeof(T), detail::getNextPowerOfTwo(sizeof(T)),
        propList,
        make_unique_ptr<detail::SYCLMemObjAllocatorHolder<AllocatorT, T>>(
            allocator));
    impl->constructorNotification(CodeLoc, (void *)impl.get(),
                                  (void *)hostData.get(),
                                  (const void *)typeid(T).name(), dimensions,
                                  sizeof(T), rangeToArray(Range).data());
  }

  buffer(const std::shared_ptr<T[]> &hostData,
         const range<dimensions> &bufferRange, AllocatorT allocator,
         const property_list &propList = {},
         const detail::code_location CodeLoc = detail::code_location::current())
      : Range(bufferRange) {
    impl = std::make_shared<detail::buffer_impl>(
        hostData, size() * sizeof(T), detail::getNextPowerOfTwo(sizeof(T)),
        propList,
        make_unique_ptr<detail::SYCLMemObjAllocatorHolder<AllocatorT, T>>(
            allocator));
    impl->constructorNotification(CodeLoc, (void *)impl.get(),
                                  (void *)hostData.get(),
                                  (const void *)typeid(T).name(), dimensions,
                                  sizeof(T), rangeToArray(Range).data());
  }

  buffer(const std::shared_ptr<T> &hostData,
         const range<dimensions> &bufferRange,
         const property_list &propList = {},
         const detail::code_location CodeLoc = detail::code_location::current())
      : Range(bufferRange) {
    impl = std::make_shared<detail::buffer_impl>(
        hostData, size() * sizeof(T), detail::getNextPowerOfTwo(sizeof(T)),
        propList,
        make_unique_ptr<detail::SYCLMemObjAllocatorHolder<AllocatorT, T>>());
    impl->constructorNotification(CodeLoc, (void *)impl.get(),
                                  (void *)hostData.get(),
                                  (const void *)typeid(T).name(), dimensions,
                                  sizeof(T), rangeToArray(Range).data());
  }

  buffer(const std::shared_ptr<T[]> &hostData,
         const range<dimensions> &bufferRange,
         const property_list &propList = {},
         const detail::code_location CodeLoc = detail::code_location::current())
      : Range(bufferRange) {
    impl = std::make_shared<detail::buffer_impl>(
        hostData, size() * sizeof(T), detail::getNextPowerOfTwo(sizeof(T)),
        propList,
        make_unique_ptr<detail::SYCLMemObjAllocatorHolder<AllocatorT, T>>());
    impl->constructorNotification(CodeLoc, (void *)impl.get(),
                                  (void *)hostData.get(),
                                  (const void *)typeid(T).name(), dimensions,
                                  sizeof(T), rangeToArray(Range).data());
  }

  template <class InputIterator, int N = dimensions,
            typename = EnableIfOneDimension<N>,
            typename = EnableIfItInputIterator<InputIterator>>
  buffer(InputIterator first, InputIterator last, AllocatorT allocator,
         const property_list &propList = {},
         const detail::code_location CodeLoc = detail::code_location::current())
      : Range(range<1>(std::distance(first, last))) {
    impl = std::make_shared<detail::buffer_impl>(
        first, last, size() * sizeof(T), detail::getNextPowerOfTwo(sizeof(T)),
        propList,
        make_unique_ptr<detail::SYCLMemObjAllocatorHolder<AllocatorT, T>>(
            allocator));
    size_t r[3] = {Range[0], 0, 0};
    impl->constructorNotification(CodeLoc, (void *)impl.get(), &first,
                                  (const void *)typeid(T).name(), dimensions,
                                  sizeof(T), r);
  }

  template <class InputIterator, int N = dimensions,
            typename = EnableIfOneDimension<N>,
            typename = EnableIfItInputIterator<InputIterator>>
  buffer(InputIterator first, InputIterator last,
         const property_list &propList = {},
         const detail::code_location CodeLoc = detail::code_location::current())
      : Range(range<1>(std::distance(first, last))) {
    impl = std::make_shared<detail::buffer_impl>(
        first, last, size() * sizeof(T), detail::getNextPowerOfTwo(sizeof(T)),
        propList,
        make_unique_ptr<detail::SYCLMemObjAllocatorHolder<AllocatorT, T>>());
    size_t r[3] = {Range[0], 0, 0};
    impl->constructorNotification(CodeLoc, (void *)impl.get(), &first,
                                  (const void *)typeid(T).name(), dimensions,
                                  sizeof(T), r);
  }

  // This constructor is a prototype for a future SYCL specification
  template <class Container, int N = dimensions,
            typename = EnableIfOneDimension<N>,
            typename = EnableIfContiguous<Container>>
  buffer(Container &container, AllocatorT allocator,
         const property_list &propList = {},
         const detail::code_location CodeLoc = detail::code_location::current())
      : Range(range<1>(container.size())) {
    impl = std::make_shared<detail::buffer_impl>(
        container.data(), size() * sizeof(T),
        detail::getNextPowerOfTwo(sizeof(T)), propList,
        make_unique_ptr<detail::SYCLMemObjAllocatorHolder<AllocatorT, T>>(
            allocator));
    size_t r[3] = {Range[0], 0, 0};
    impl->constructorNotification(CodeLoc, (void *)impl.get(), container.data(),
                                  (const void *)typeid(T).name(), dimensions,
                                  sizeof(T), r);
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
      : impl(b.impl), Range(subRange),
        OffsetInBytes(getOffsetInBytes<T>(baseIndex, b.Range)),
        IsSubBuffer(true) {
    impl->constructorNotification(CodeLoc, (void *)impl.get(), impl.get(),
                                  (const void *)typeid(T).name(), dimensions,
                                  sizeof(T), rangeToArray(Range).data());

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

#ifdef __SYCL_INTERNAL_API
  template <int N = dimensions, typename = EnableIfOneDimension<N>>
  buffer(cl_mem MemObject, const context &SyclContext,
         event AvailableEvent = {},
         const detail::code_location CodeLoc = detail::code_location::current())
      : Range{0} {

    impl = std::make_shared<detail::buffer_impl>(
        detail::pi::cast<pi_native_handle>(MemObject), SyclContext,
        make_unique_ptr<detail::SYCLMemObjAllocatorHolder<AllocatorT, T>>(),
        /* OwnNativeHandle */ true, AvailableEvent);
    Range[0] = impl->getSize() / sizeof(T);
    impl->constructorNotification(CodeLoc, (void *)impl.get(), &MemObject,
                                  (const void *)typeid(T).name(), dimensions,
                                  sizeof(T), rangeToArray(Range).data());
  }
#endif

  buffer(const buffer &rhs,
         const detail::code_location CodeLoc = detail::code_location::current())
      : impl(rhs.impl), Range(rhs.Range), OffsetInBytes(rhs.OffsetInBytes),
        IsSubBuffer(rhs.IsSubBuffer) {
    impl->constructorNotification(CodeLoc, (void *)impl.get(), impl.get(),
                                  (const void *)typeid(T).name(), dimensions,
                                  sizeof(T), rangeToArray(Range).data());
  }

  buffer(buffer &&rhs,
         const detail::code_location CodeLoc = detail::code_location::current())
      : impl(std::move(rhs.impl)), Range(rhs.Range),
        OffsetInBytes(rhs.OffsetInBytes), IsSubBuffer(rhs.IsSubBuffer) {
    impl->constructorNotification(CodeLoc, (void *)impl.get(), impl.get(),
                                  (const void *)typeid(T).name(), dimensions,
                                  sizeof(T), rangeToArray(Range).data());
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
    return impl->template get_allocator<AllocatorT>();
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
    impl->set_final_data(finalData);
  }

  void set_write_back(bool flag = true) { impl->set_write_back(flag); }

  bool is_sub_buffer() const { return IsSubBuffer; }

  template <typename ReinterpretT, int ReinterpretDim>
  buffer<ReinterpretT, ReinterpretDim,
         typename std::allocator_traits<AllocatorT>::template rebind_alloc<
             ReinterpretT>>
  reinterpret(range<ReinterpretDim> reinterpretRange) const {
    if (sizeof(ReinterpretT) * reinterpretRange.size() != byte_size())
      throw sycl::invalid_object_error(
          "Total size in bytes represented by the type and range of the "
          "reinterpreted SYCL buffer does not equal the total size in bytes "
          "represented by the type and range of this SYCL buffer",
          PI_ERROR_INVALID_VALUE);

    return buffer<ReinterpretT, ReinterpretDim,
                  typename std::allocator_traits<
                      AllocatorT>::template rebind_alloc<ReinterpretT>>(
        impl, reinterpretRange, OffsetInBytes, IsSubBuffer);
  }

  template <typename ReinterpretT, int ReinterpretDim = dimensions>
  typename std::enable_if<
      (sizeof(ReinterpretT) == sizeof(T)) && (dimensions == ReinterpretDim),
      buffer<ReinterpretT, ReinterpretDim,
             typename std::allocator_traits<AllocatorT>::template rebind_alloc<
                 ReinterpretT>>>::type
  reinterpret() const {
    return buffer<ReinterpretT, ReinterpretDim,
                  typename std::allocator_traits<
                      AllocatorT>::template rebind_alloc<ReinterpretT>>(
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
    return impl->template has_property<propertyT>();
  }

  template <typename propertyT> propertyT get_property() const {
    return impl->template get_property<propertyT>();
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
  std::shared_ptr<detail::buffer_impl> impl;
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
      : Range{0} {

    impl = std::make_shared<detail::buffer_impl>(
        MemObject, SyclContext,
        make_unique_ptr<detail::SYCLMemObjAllocatorHolder<AllocatorT, T>>(),
        OwnNativeHandle, AvailableEvent);
    Range[0] = impl->getSize() / sizeof(T);
    impl->constructorNotification(CodeLoc, (void *)impl.get(), &MemObject,
                                  (const void *)typeid(T).name(), dimensions,
                                  sizeof(T), rangeToArray(Range).data());
  }

  // Reinterpret contructor
  buffer(std::shared_ptr<detail::buffer_impl> Impl,
         range<dimensions> reinterpretRange, size_t reinterpretOffset,
         bool isSubBuffer,
         const detail::code_location CodeLoc = detail::code_location::current())
      : impl(Impl), Range(reinterpretRange), OffsetInBytes(reinterpretOffset),
        IsSubBuffer(isSubBuffer) {
    impl->constructorNotification(CodeLoc, (void *)impl.get(), Impl.get(),
                                  (const void *)typeid(T).name(), dimensions,
                                  sizeof(T), rangeToArray(Range).data());
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
    auto NativeHandles = impl->getNativeVector(BackendName);
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
