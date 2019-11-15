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
#include <CL/sycl/exception.hpp>
#include <CL/sycl/stl.hpp>

// TODO: 4.3.4 Properties

namespace cl {
namespace sycl {
class handler;
class queue;
template <int dimensions> class range;

template <typename T, int dimensions = 1,
          typename AllocatorT = cl::sycl::buffer_allocator>
class buffer {
public:
  using value_type = T;
  using reference = value_type &;
  using const_reference = const value_type &;
  using allocator_type = AllocatorT;
  template <int dims>
  using EnableIfOneDimension = typename std::enable_if<1 == dims>::type;
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
  using EnableIfSameNonConstIterators =
      typename std::enable_if<std::is_same<ItA, ItB>::value &&
                              !std::is_const<ItA>::value, ItA>::type;

  buffer(const range<dimensions> &bufferRange,
         const property_list &propList = {})
      : Range(bufferRange) {
    impl = std::make_shared<detail::buffer_impl<AllocatorT>>(
        get_count() * sizeof(T), detail::getNextPowerOfTwo(sizeof(T)),
        propList);
  }

  buffer(const range<dimensions> &bufferRange, AllocatorT allocator,
         const property_list &propList = {})
      : Range(bufferRange) {
    impl = std::make_shared<detail::buffer_impl<AllocatorT>>(
        get_count() * sizeof(T), detail::getNextPowerOfTwo(sizeof(T)), propList,
        allocator);
  }

  buffer(T *hostData, const range<dimensions> &bufferRange,
         const property_list &propList = {})
      : Range(bufferRange) {
    impl = std::make_shared<detail::buffer_impl<AllocatorT>>(
        hostData, get_count() * sizeof(T), detail::getNextPowerOfTwo(sizeof(T)),
        propList);
  }

  buffer(T *hostData, const range<dimensions> &bufferRange,
         AllocatorT allocator, const property_list &propList = {})
      : Range(bufferRange) {
    impl = std::make_shared<detail::buffer_impl<AllocatorT>>(
        hostData, get_count() * sizeof(T), detail::getNextPowerOfTwo(sizeof(T)),
        propList, allocator);
  }

  template <typename _T = T>
  buffer(EnableIfSameNonConstIterators<T, _T> const *hostData,
         const range<dimensions> &bufferRange,
         const property_list &propList = {})
      : Range(bufferRange) {
    impl = std::make_shared<detail::buffer_impl<AllocatorT>>(
        hostData, get_count() * sizeof(T), detail::getNextPowerOfTwo(sizeof(T)),
        propList);
  }

  template <typename _T = T>
  buffer(EnableIfSameNonConstIterators<T, _T> const *hostData,
         const range<dimensions> &bufferRange, AllocatorT allocator,
         const property_list &propList = {})
      : Range(bufferRange) {
    impl = std::make_shared<detail::buffer_impl<AllocatorT>>(
        hostData, get_count() * sizeof(T), detail::getNextPowerOfTwo(sizeof(T)),
        propList, allocator);
  }

  buffer(const shared_ptr_class<T> &hostData,
         const range<dimensions> &bufferRange, AllocatorT allocator,
         const property_list &propList = {})
      : Range(bufferRange) {
    impl = std::make_shared<detail::buffer_impl<AllocatorT>>(
        hostData, get_count() * sizeof(T), detail::getNextPowerOfTwo(sizeof(T)),
        propList, allocator);
  }

  buffer(const shared_ptr_class<T> &hostData,
         const range<dimensions> &bufferRange,
         const property_list &propList = {})
      : Range(bufferRange) {
    impl = std::make_shared<detail::buffer_impl<AllocatorT>>(
        hostData, get_count() * sizeof(T), detail::getNextPowerOfTwo(sizeof(T)),
        propList);
  }

  template <class InputIterator, int N = dimensions,
            typename = EnableIfOneDimension<N>,
            typename = EnableIfItInputIterator<InputIterator>>
  buffer(InputIterator first, InputIterator last, AllocatorT allocator,
         const property_list &propList = {})
      : Range(range<1>(std::distance(first, last))) {
    impl = std::make_shared<detail::buffer_impl<AllocatorT>>(
        first, last, get_count() * sizeof(T),
        detail::getNextPowerOfTwo(sizeof(T)), propList, allocator);
  }

  template <class InputIterator, int N = dimensions,
            typename = EnableIfOneDimension<N>,
            typename = EnableIfItInputIterator<InputIterator>>
  buffer(InputIterator first, InputIterator last,
         const property_list &propList = {})
      : Range(range<1>(std::distance(first, last))) {
    impl = std::make_shared<detail::buffer_impl<AllocatorT>>(
        first, last, get_count() * sizeof(T),
        detail::getNextPowerOfTwo(sizeof(T)), propList);
  }

  // This constructor is a prototype for a future SYCL specification
  template <class Container, int N = dimensions,
            typename = EnableIfOneDimension<N>,
            typename = EnableIfContiguous<Container>>
  buffer(Container &container, AllocatorT allocator,
         const property_list &propList = {})
      : Range(range<1>(container.size())) {
    impl = std::make_shared<detail::buffer_impl<AllocatorT>>(
        container.data(), container.data() + container.size(),
        get_count() * sizeof(T), detail::getNextPowerOfTwo(sizeof(T)), propList,
        allocator);
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
          "Cannot create sub buffer from sub buffer.");
    if (isOutOfBounds(baseIndex, subRange, b.Range))
      throw cl::sycl::invalid_object_error(
          "Requested sub-buffer size exceeds the size of the parent buffer");
    if (!isContiguousRegion(baseIndex, subRange, b.Range))
      throw cl::sycl::invalid_object_error(
          "Requested sub-buffer region is not contiguous");
  }

  template <int N = dimensions, typename = EnableIfOneDimension<N>>
  buffer(cl_mem MemObject, const context &SyclContext,
         event AvailableEvent = {})
      : Range{0} {

    size_t BufSize = 0;
    PI_CALL(detail::RT::piMemGetInfo,
        detail::pi::cast<detail::RT::PiMem>(MemObject), CL_MEM_SIZE,
        sizeof(size_t), &BufSize, nullptr);

    Range[0] = BufSize / sizeof(T);
    impl = std::make_shared<detail::buffer_impl<AllocatorT>>(
        MemObject, SyclContext, BufSize, AvailableEvent);
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

  size_t get_count() const { return Range.size(); }

  size_t get_size() const { return get_count() * sizeof(T); }

  AllocatorT get_allocator() const { return impl->get_allocator(); }

  template <access::mode mode,
            access::target target = access::target::global_buffer>
  accessor<T, dimensions, mode, target, access::placeholder::false_t>
  get_access(handler &commandGroupHandler) {
    return impl->template get_access<T, dimensions, mode, target>(
        *this, commandGroupHandler);
  }

  template <access::mode mode>
  accessor<T, dimensions, mode, access::target::host_buffer,
           access::placeholder::false_t>
  get_access() {
    return impl->template get_access<T, dimensions, mode>(*this);
  }

  template <access::mode mode,
            access::target target = access::target::global_buffer>
  accessor<T, dimensions, mode, target, access::placeholder::false_t>
  get_access(handler &commandGroupHandler, range<dimensions> accessRange,
             id<dimensions> accessOffset = {}) {
    return impl->template get_access<T, dimensions, mode, target>(
        *this, commandGroupHandler, accessRange, accessOffset);
  }

  template <access::mode mode>
  accessor<T, dimensions, mode, access::target::host_buffer,
           access::placeholder::false_t>
  get_access(range<dimensions> accessRange, id<dimensions> accessOffset = {}) {
    return impl->template get_access<T, dimensions, mode>(*this, accessRange,
                                                          accessOffset);
  }

  template <typename Destination = std::nullptr_t>
  void set_final_data(Destination finalData = nullptr) {
    impl->set_final_data(finalData);
  }

  void set_write_back(bool flag = true) { impl->set_write_back(flag); }

  bool is_sub_buffer() const { return IsSubBuffer; }

  template <typename ReinterpretT, int ReinterpretDim>
  buffer<ReinterpretT, ReinterpretDim, AllocatorT>
  reinterpret(range<ReinterpretDim> reinterpretRange) const {
    if (sizeof(ReinterpretT) * reinterpretRange.size() != get_size())
      throw cl::sycl::invalid_object_error(
          "Total size in bytes represented by the type and range of the "
          "reinterpreted SYCL buffer does not equal the total size in bytes "
          "represented by the type and range of this SYCL buffer");

    return buffer<ReinterpretT, ReinterpretDim, AllocatorT>(
        impl, reinterpretRange, OffsetInBytes, IsSubBuffer);
  }

  template <typename propertyT> bool has_property() const {
    return impl->template has_property<propertyT>();
  }

  template <typename propertyT> propertyT get_property() const {
    return impl->template get_property<propertyT>();
  }

private:
  shared_ptr_class<detail::buffer_impl<AllocatorT>> impl;
  template <class Obj>
  friend decltype(Obj::impl) detail::getSyclObjImpl(const Obj &SyclObject);
  template <typename A, int dims, typename C> friend class buffer;
  template <typename DataT, int dims, access::mode mode,
            access::target target, access::placeholder isPlaceholder>
  friend class accessor;
  range<dimensions> Range;
  // Offset field specifies the origin of the sub buffer inside the parent
  // buffer
  size_t OffsetInBytes = 0;
  bool IsSubBuffer = false;

  // Reinterpret contructor
  buffer(shared_ptr_class<detail::buffer_impl<AllocatorT>> Impl,
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

  bool isContiguousRegion(const id<1> &offset, const range<1> &newRange,
                          const range<1> &parentRange) {
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
    ->buffer<typename std::iterator_traits<InputIterator>::value_type, 1,
             AllocatorT>;
template <class InputIterator>
buffer(InputIterator, InputIterator, const property_list & = {})
    ->buffer<typename std::iterator_traits<InputIterator>::value_type, 1>;
template <class Container, class AllocatorT>
buffer(Container &, AllocatorT, const property_list & = {})
    ->buffer<typename Container::value_type, 1, AllocatorT>;
template <class Container>
buffer(Container &, const property_list & = {})
    ->buffer<typename Container::value_type, 1>;
template <class T, int dimensions, class AllocatorT>
buffer(const T *, const range<dimensions> &, AllocatorT,
       const property_list & = {})
    ->buffer<T, dimensions, AllocatorT>;
template <class T, int dimensions>
buffer(const T *, const range<dimensions> &, const property_list & = {})
    ->buffer<T, dimensions>;
#endif // __cpp_deduction_guides

} // namespace sycl
} // namespace cl

namespace std {
template <typename T, int dimensions, typename AllocatorT>
struct hash<cl::sycl::buffer<T, dimensions, AllocatorT>> {
  size_t
  operator()(const cl::sycl::buffer<T, dimensions, AllocatorT> &b) const {
    return hash<std::shared_ptr<cl::sycl::detail::buffer_impl<AllocatorT>>>()(
        cl::sycl::detail::getSyclObjImpl(b));
  }
};
} // namespace std
