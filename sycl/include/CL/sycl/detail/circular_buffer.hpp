//==---------------- circular_buffer.hpp - Circular buffer -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/defines.hpp>

#include <cstddef>
#include <deque>
#include <utility>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

// A partial implementation of a circular buffer: once its capacity is full,
// new data overwrites the old.
template <typename T> class CircularBuffer {
public:
  explicit CircularBuffer(std::size_t Capacity) : MCapacity{Capacity} {};

  using value_type = T;
  using pointer = T *;
  using const_pointer = const T *;
  using reference = T &;
  using const_reference = const T &;

  using iterator = typename std::deque<T>::iterator;
  using const_iterator = typename std::deque<T>::const_iterator;

  iterator begin() { return MValues.begin(); }

  const_iterator begin() const { return MValues.begin(); }

  iterator end() { return MValues.end(); }

  const_iterator end() const { return MValues.end(); }

  reference front() { return MValues.front(); }

  const_reference front() const { return MValues.front(); }

  reference back() { return MValues.back(); }

  const_reference back() const { return MValues.back(); }

  reference operator[](std::size_t Idx) { return MValues[Idx]; }

  const_reference operator[](std::size_t Idx) const { return MValues[Idx]; }

  std::size_t size() const { return MValues.size(); }

  std::size_t capacity() const { return MCapacity; }

  bool empty() const { return MValues.empty(); };

  bool full() const { return MValues.size() == MCapacity; };

  void push_back(T Val) {
    if (MValues.size() == MCapacity)
      MValues.pop_front();
    MValues.push_back(std::move(Val));
  }

  void push_front(T Val) {
    if (MValues.size() == MCapacity)
      MValues.pop_back();
    MValues.push_front(std::move(Val));
  }

  void pop_back() { MValues.pop_back(); }

  void pop_front() { MValues.pop_front(); }

  void erase(const_iterator Pos) { MValues.erase(Pos); }

  void erase(const_iterator First, const_iterator Last) {
    MValues.erase(First, Last);
  }

  void clear() { MValues.clear(); }

private:
  // Deque is used as the underlying container for double-ended push/pop
  // operations and built-in iterator support. Frequent memory allocations
  // and deallocations are a concern, switching to an array/vector might be a
  // worthwhile optimization.
  std::deque<T> MValues;
  const std::size_t MCapacity;
};

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
