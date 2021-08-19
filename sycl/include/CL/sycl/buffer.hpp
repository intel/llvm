//==----------- buffer.hpp --- SYCL buffer ---------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/buffer_impl.hpp>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/stl_type_traits.hpp>
#include <CL/sycl/exception.hpp>
#include <CL/sycl/property_list.hpp>
#include <CL/sycl/stl.hpp>
#include <sycl/ext/oneapi/accessor_property_list.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

class handler;
class queue;
template <int dimensions> class range;

/// Defines a shared array that can be used by kernels in queues.
///
/// Buffers can be 1-, 2-, and 3-dimensional. They have to be accessed using
/// accessor classes.
///
/// \sa sycl_api_acc
///
/// \ingroup sycl_api
template <typename T, int dimensions = 1,
          typename AllocatorT = cl::sycl::buffer_allocator,
          typename = typename detail::enable_if_t<(dimensions > 0) &&
                                                  (dimensions <= 3)>>
class buffer {
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
                         detail::remove_pointer_t<decltype(
                             std::declval<Container>().data())> (*)[],
                         const T (*)[]>::value>,
                     decltype(std::declval<Container>().size())>;
  template <class It>
  using EnableIfItInputIterator = detail::enable_if_t<
      std::is_convertible<typename std::iterator_traits<It>::iterator_category,
                          std::input_iterator_tag>::value>;
  template <typename ItA, typename ItB>
  using EnableIfSameNonConstIterators = typename detail::enable_if_t<
      std::is_same<ItA, ItB>::value && !std::is_const<ItA>::value, ItA>;

  buffer(const range<dimensions> &bufferRange,
         const property_list &propList = {})
      : Range(bufferRange) {
    impl = std::make_shared<detail::buffer_impl>(
        size() * sizeof(T), detail::getNextPowerOfTwo(sizeof(T)), propList,
        make_unique_ptr<detail::SYCLMemObjAllocatorHolder<AllocatorT>>());
  }

  buffer(const range<dimensions> &bufferRange, AllocatorT allocator,
         const property_list &propList = {})
      : Range(bufferRange) {
    impl = std::make_shared<detail::buffer_impl>(
        size() * sizeof(T), detail::getNextPowerOfTwo(sizeof(T)), propList,
        make_unique_ptr<detail::SYCLMemObjAllocatorHolder<AllocatorT>>(
            allocator));
  }

  buffer(T *hostData, const range<dimensions> &bufferRange,
         const property_list &propList = {})
      : Range(bufferRange) {
    impl = std::make_shared<detail::buffer_impl>(
        hostData, size() * sizeof(T), detail::getNextPowerOfTwo(sizeof(T)),
        propList,
        make_unique_ptr<detail::SYCLMemObjAllocatorHolder<AllocatorT>>());
  }

  buffer(T *hostData, const range<dimensions> &bufferRange,
         AllocatorT allocator, const property_list &propList = {})
      : Range(bufferRange) {
    impl = std::make_shared<detail::buffer_impl>(
        hostData, size() * sizeof(T), detail::getNextPowerOfTwo(sizeof(T)),
        propList,
        make_unique_ptr<detail::SYCLMemObjAllocatorHolder<AllocatorT>>(
            allocator));
  }

  template <typename _T = T>
  buffer(EnableIfSameNonConstIterators<T, _T> const *hostData,
         const range<dimensions> &bufferRange,
         const property_list &propList = {})
      : Range(bufferRange) {
    impl = std::make_shared<detail::buffer_impl>(
        hostData, size() * sizeof(T), detail::getNextPowerOfTwo(sizeof(T)),
        propList,
        make_unique_ptr<detail::SYCLMemObjAllocatorHolder<AllocatorT>>());
  }

  template <typename _T = T>
  buffer(EnableIfSameNonConstIterators<T, _T> const *hostData,
         const range<dimensions> &bufferRange, AllocatorT allocator,
         const property_list &propList = {})
      : Range(bufferRange) {
    impl = std::make_shared<detail::buffer_impl>(
        hostData, size() * sizeof(T), detail::getNextPowerOfTwo(sizeof(T)),
        propList,
        make_unique_ptr<detail::SYCLMemObjAllocatorHolder<AllocatorT>>(
            allocator));
  }

  buffer(const std::shared_ptr<T> &hostData,
         const range<dimensions> &bufferRange, AllocatorT allocator,
         const property_list &propList = {})
      : Range(bufferRange) {
    impl = std::make_shared<detail::buffer_impl>(
        hostData, size() * sizeof(T), detail::getNextPowerOfTwo(sizeof(T)),
        propList,
        make_unique_ptr<detail::SYCLMemObjAllocatorHolder<AllocatorT>>(
            allocator));
  }

  buffer(const std::shared_ptr<T[]> &hostData,
         const range<dimensions> &bufferRange, AllocatorT allocator,
         const property_list &propList = {})
      : Range(bufferRange) {
    impl = std::make_shared<detail::buffer_impl>(
        hostData, size() * sizeof(T), detail::getNextPowerOfTwo(sizeof(T)),
        propList,
        make_unique_ptr<detail::SYCLMemObjAllocatorHolder<AllocatorT>>(
            allocator));
  }

  buffer(const std::shared_ptr<T> &hostData,
         const range<dimensions> &bufferRange,
         const property_list &propList = {})
      : Range(bufferRange) {
    impl = std::make_shared<detail::buffer_impl>(
        hostData, size() * sizeof(T), detail::getNextPowerOfTwo(sizeof(T)),
        propList,
        make_unique_ptr<detail::SYCLMemObjAllocatorHolder<AllocatorT>>());
  }

  buffer(const std::shared_ptr<T[]> &hostData,
         const range<dimensions> &bufferRange,
         const property_list &propList = {})
      : Range(bufferRange) {
    impl = std::make_shared<detail::buffer_impl>(
        hostData, size() * sizeof(T), detail::getNextPowerOfTwo(sizeof(T)),
        propList,
        make_unique_ptr<detail::SYCLMemObjAllocatorHolder<AllocatorT>>());
  }

  template <class InputIterator, int N = dimensions,
            typename = EnableIfOneDimension<N>,
            typename = EnableIfItInputIterator<InputIterator>>
  buffer(InputIterator first, InputIterator last, AllocatorT allocator,
         const property_list &propList = {})
      : Range(range<1>(std::distance(first, last))) {
    impl = std::make_shared<detail::buffer_impl>(
        first, last, size() * sizeof(T), detail::getNextPowerOfTwo(sizeof(T)),
        propList,
        make_unique_ptr<detail::SYCLMemObjAllocatorHolder<AllocatorT>>(
            allocator));
  }

  template <class InputIterator, int N = dimensions,
            typename = EnableIfOneDimension<N>,
            typename = EnableIfItInputIterator<InputIterator>>
  buffer(InputIterator first, InputIterator last,
         const property_list &propList = {})
      : Range(range<1>(std::distance(first, last))) {
    impl = std::make_shared<detail::buffer_impl>(
        first, last, size() * sizeof(T), detail::getNextPowerOfTwo(sizeof(T)),
        propList,
        make_unique_ptr<detail::SYCLMemObjAllocatorHolder<AllocatorT>>());
  }

  // This constructor is a prototype for a future SYCL specification
  template <class Container, int N = dimensions,
            typename = EnableIfOneDimension<N>,
            typename = EnableIfContiguous<Container>>
  buffer(Container &container, AllocatorT allocator,
         const property_list &propList = {})
      : Range(range<1>(container.size())) {
    impl = std::make_shared<detail::buffer_impl>(
        container.data(), size() * sizeof(T),
        detail::getNextPowerOfTwo(sizeof(T)), propList,
        make_unique_ptr<detail::SYCLMemObjAllocatorHolder<AllocatorT>>(
            allocator));
  }

  // This constructor is a prototype for a future SYCL specification
  template <class Container, int N = dimensions,
            typename = EnableIfOneDimension<N>,
            typename = EnableIfContiguous<Container>>
  buffer(Container &container, const property_list &propList = {})
      : buffer(container, {}, propList) {}

  buffer(buffer<T, dimensions, AllocatorT> &b, const id<dimensions> &baseIndex,
         const range<dimensions> &subRange)
      : impl(b.impl), Range(subRange),
        OffsetInBytes(getOffsetInBytes<T>(baseIndex, b.Range)),
        IsSubBuffer(true) {
    if (b.is_sub_buffer())
      throw cl::sycl::invalid_object_error(
          "Cannot create sub buffer from sub buffer.", PI_INVALID_VALUE);
    if (isOutOfBounds(baseIndex, subRange, b.Range))
      throw cl::sycl::invalid_object_error(
          "Requested sub-buffer size exceeds the size of the parent buffer",
          PI_INVALID_VALUE);
    if (!isContiguousRegion(baseIndex, subRange, b.Range))
      throw cl::sycl::invalid_object_error(
          "Requested sub-buffer region is not contiguous", PI_INVALID_VALUE);
  }

  template <int N = dimensions, typename = EnableIfOneDimension<N>>
  __SYCL2020_DEPRECATED("OpenCL interop APIs are deprecated")
  buffer(cl_mem MemObject, const context &SyclContext,
         event AvailableEvent = {})
      : Range{0} {

    size_t BufSize = detail::SYCLMemObjT::getBufSizeForContext(
        detail::getSyclObjImpl(SyclContext), MemObject);

    Range[0] = BufSize / sizeof(T);
    impl = std::make_shared<detail::buffer_impl>(
        MemObject, SyclContext, BufSize,
        make_unique_ptr<detail::SYCLMemObjAllocatorHolder<AllocatorT>>(),
        AvailableEvent);
  }

  buffer(const buffer &rhs) = default;

  buffer(buffer &&rhs) = default;

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

  template <access::mode Mode,
            access::target Target = access::target::global_buffer>
  accessor<T, dimensions, Mode, Target, access::placeholder::false_t,
           ext::oneapi::accessor_property_list<>>
  get_access(handler &CommandGroupHandler) {
    return accessor<T, dimensions, Mode, Target, access::placeholder::false_t,
                    ext::oneapi::accessor_property_list<>>(*this,
                                                           CommandGroupHandler);
  }

  template <access::mode mode>
  accessor<T, dimensions, mode, access::target::host_buffer,
           access::placeholder::false_t, ext::oneapi::accessor_property_list<>>
  get_access() {
    return accessor<T, dimensions, mode, access::target::host_buffer,
                    access::placeholder::false_t,
                    ext::oneapi::accessor_property_list<>>(*this);
  }

  template <access::mode mode,
            access::target target = access::target::global_buffer>
  accessor<T, dimensions, mode, target, access::placeholder::false_t,
           ext::oneapi::accessor_property_list<>>
  get_access(handler &commandGroupHandler, range<dimensions> accessRange,
             id<dimensions> accessOffset = {}) {
    return accessor<T, dimensions, mode, target, access::placeholder::false_t,
                    ext::oneapi::accessor_property_list<>>(
        *this, commandGroupHandler, accessRange, accessOffset);
  }

  template <access::mode mode>
  accessor<T, dimensions, mode, access::target::host_buffer,
           access::placeholder::false_t, ext::oneapi::accessor_property_list<>>
  get_access(range<dimensions> accessRange, id<dimensions> accessOffset = {}) {
    return accessor<T, dimensions, mode, access::target::host_buffer,
                    access::placeholder::false_t,
                    ext::oneapi::accessor_property_list<>>(*this, accessRange,
                                                           accessOffset);
  }

#if __cplusplus > 201402L

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
  buffer<ReinterpretT, ReinterpretDim, AllocatorT>
  reinterpret(range<ReinterpretDim> reinterpretRange) const {
    if (sizeof(ReinterpretT) * reinterpretRange.size() != byte_size())
      throw cl::sycl::invalid_object_error(
          "Total size in bytes represented by the type and range of the "
          "reinterpreted SYCL buffer does not equal the total size in bytes "
          "represented by the type and range of this SYCL buffer",
          PI_INVALID_VALUE);

    return buffer<ReinterpretT, ReinterpretDim, AllocatorT>(
        impl, reinterpretRange, OffsetInBytes, IsSubBuffer);
  }

  template <typename ReinterpretT, int ReinterpretDim = dimensions>
  typename std::enable_if<
      (sizeof(ReinterpretT) == sizeof(T)) && (dimensions == ReinterpretDim),
      buffer<ReinterpretT, ReinterpretDim, AllocatorT>>::type
  reinterpret() const {
    return buffer<ReinterpretT, ReinterpretDim, AllocatorT>(
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
      throw cl::sycl::invalid_object_error(
          "Total byte size of buffer is not evenly divisible by the size of "
          "the reinterpreted type",
          PI_INVALID_VALUE);

    return buffer<ReinterpretT, ReinterpretDim, AllocatorT>(
        impl, range<1>{sz / sizeof(ReinterpretT)}, OffsetInBytes, IsSubBuffer);
  }

  template <typename propertyT> bool has_property() const {
    return impl->template has_property<propertyT>();
  }

  template <typename propertyT> propertyT get_property() const {
    return impl->template get_property<propertyT>();
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
  range<dimensions> Range;
  // Offset field specifies the origin of the sub buffer inside the parent
  // buffer
  size_t OffsetInBytes = 0;
  bool IsSubBuffer = false;

  // Reinterpret contructor
  buffer(std::shared_ptr<detail::buffer_impl> Impl,
         range<dimensions> reinterpretRange, size_t reinterpretOffset,
         bool isSubBuffer)
      : impl(Impl), Range(reinterpretRange), OffsetInBytes(reinterpretOffset),
        IsSubBuffer(isSubBuffer){};

  template <typename Type, int N>
  size_t getOffsetInBytes(const id<N> &offset, const range<N> &range) {
    return detail::getLinearIndex(offset, range) * sizeof(Type);
  }

  bool isOutOfBounds(const id<dimensions> &offset,
                     const range<dimensions> &newRange,
                     const range<dimensions> &parentRange) {
    bool outOfBounds = false;
    for (int i = 0; i < dimensions; ++i)
      outOfBounds |= newRange[i] + offset[i] > parentRange[i];

    return outOfBounds;
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

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

namespace std {
template <typename T, int dimensions, typename AllocatorT>
struct hash<cl::sycl::buffer<T, dimensions, AllocatorT>> {
  size_t
  operator()(const cl::sycl::buffer<T, dimensions, AllocatorT> &b) const {
    return hash<std::shared_ptr<cl::sycl::detail::buffer_impl>>()(
        cl::sycl::detail::getSyclObjImpl(b));
  }
};
} // namespace std
