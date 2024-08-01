//==------------ DynArray.h - Non-STL replacement for std::array -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SYCL_FUSION_COMMON_DYNARRAY_H
#define SYCL_FUSION_COMMON_DYNARRAY_H

#include <algorithm>

namespace jit_compiler {

///
/// A fixed-size, dynamically-allocated array, with an interface that is a
/// subset of `std::array`.
template <typename T> class DynArray {
public:
  DynArray() = default;

  explicit DynArray(size_t Size) { init(Size); }

  ~DynArray() { deinit(); }

  DynArray(const DynArray &Other) {
    init(Other.Size);
    std::copy(Other.begin(), Other.end(), begin());
  }

  DynArray &operator=(const DynArray &Other) {
    deinit();
    init(Other.Size);
    std::copy(Other.begin(), Other.end(), begin());
    return *this;
  }

  DynArray(DynArray &&Other) { moveFrom(std::move(Other)); }

  DynArray &operator=(DynArray &&Other) {
    deinit();
    moveFrom(std::move(Other));
    return *this;
  }

  size_t size() const { return Size; }
  bool empty() const { return Size == 0; }

  const T *begin() const { return Values; }
  const T *end() const { return Values + Size; }
  T *begin() { return Values; }
  T *end() { return Values + Size; }

  const T &operator[](int Idx) const { return Values[Idx]; }
  T &operator[](int Idx) { return Values[Idx]; }

  friend bool operator==(const DynArray<T> &A, const DynArray<T> &B) {
    return std::equal(A.begin(), A.end(), B.begin(), B.end());
  }

  friend bool operator!=(const DynArray<T> &A, const DynArray<T> &B) {
    return !(A == B);
  }

private:
  T *Values = nullptr;
  size_t Size = 0;

  void init(size_t NewSize) {
    if (NewSize == 0)
      return;

    Values = new T[NewSize];
    Size = NewSize;
  }

  void deinit() {
    if (!Values)
      return;

    delete[] Values;
    Values = nullptr;
    Size = 0;
  }

  void moveFrom(DynArray &&Other) {
    Values = Other.Values;
    Other.Values = nullptr;
    Size = Other.Size;
    Other.Size = 0;
  }
};

} // namespace jit_compiler

#endif // SYCL_FUSION_COMMON_DYNARRAY_H
