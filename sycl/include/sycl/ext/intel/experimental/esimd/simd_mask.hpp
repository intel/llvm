//==-------------- simd_mask.hpp - DPC++ Explicit SIMD API -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// simd_mask class definition - represents Gen simd operation mask.
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/defines.hpp>
#include <CL/sycl/detail/stl_type_traits.hpp> // to define C++14,17 extensions
#include <sycl/ext/intel/experimental/esimd/detail/esimd_memory_intrin.hpp>
#include <sycl/ext/intel/experimental/esimd/detail/esimd_types.hpp>

#include <cstdint>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace intel {
namespace experimental {
namespace esimd {

// Represents a Gen simd operation mask. Aims to have similar interface to
// std::experimantal::simd_mask:
// (https://github.com/llvm/llvm-project/blob/main/libcxx/include/experimental/simd)
// Does not provide the following members/features:
// - abi_type (not needed for ESIMD)
// - simd_type (not needed for ESIMD)
// - reference (TODO)
// - implicit type conversion constructor (not needed for ESIMD)
// - reference operator[](size_t) (TODO)
// - flags in memory load/store operations (TODO)
// - reduction functions (TODO):
//     any_of, all_of, some_of, none_of, popcount, find_first_set, find_last_set
template <int N> class simd_mask {
private:
  using simd_mask_impl_t = typename detail::simd_mask_impl_t<N>;

public:
  using value_type = typename detail::simd_mask_impl<N>::value_type;

  static constexpr size_t size() noexcept { return N; }

  // Default constructor.
  simd_mask() = default;

  /// Broadcast constructor.
  /// NOTE: std::exprimental::simd_mask provides broadcast constructor only for
  /// the value_type argument.
  template <typename T, typename = std::enable_if_t<std::is_integral<T>::value>>
  simd_mask(T v) noexcept {
    for (auto I = 0; I < N; ++I) {
      Val[I] = v;
    }
  }

  // TODO add accessor-based mask memory operations.

  /// Load constructor.
  // Implementation note: use SFINAE to avoid overload ambiguity:
  // 1) with 'simd_mask(value_type v)' in 'simd_mask<N> m(0)'
  // 2) with 'simd_mask(const T1(&&arr)[N])' in simd_mask<N> m((value_type*)p)'
  template <typename T,
            typename = std::enable_if_t<std::is_same<T, value_type>::value>>
  explicit simd_mask(const T *ptr) {
    copy_from(ptr);
  }

#define __ESIMD_UNSUPPORTED_MASK_SIZE_MSG                                      \
  "This sime_mask size is not supported yet in memory I/O operations. "        \
  "Supported sizes are 8, 18, 32."

private:
  static inline constexpr bool mask_size_ok_for_mem_io() {
    constexpr unsigned Sz = sizeof(value_type) * N;
    return (Sz >= detail::OperandSize::OWORD) &&
           (Sz % detail::OperandSize::OWORD == 0) &&
           detail::isPowerOf2(Sz / detail::OperandSize::OWORD) &&
           (Sz <= 8 * detail::OperandSize::OWORD);
  }

public:
  /// Load the mask's value from memory.
  /// TODO support arbitrary sizes.
  template <typename T,
            typename = std::enable_if_t<std::is_same<T, value_type>::value>>
  void copy_from(const T *ptr) {
    static_assert(mask_size_ok_for_mem_io(), __ESIMD_UNSUPPORTED_MASK_SIZE_MSG);
    set(__esimd_flat_block_read_unaligned<value_type, N>(
        reinterpret_cast<uintptr_t>(ptr)));
  }

  /// Store the mask's value to memory.
  template <typename T,
            typename = std::enable_if_t<std::is_same<T, value_type>::value>>
  void copy_to(T *ptr) const {
    static_assert(mask_size_ok_for_mem_io(), __ESIMD_UNSUPPORTED_MASK_SIZE_MSG);
    __esimd_flat_block_write<T, N>(reinterpret_cast<uintptr_t>(ptr), data());
  }

#undef __ESIMD_UNSUPPORTED_MASK_SIZE_MSG

  value_type operator[](size_t i) const { return data()[i]; }

  simd_mask operator!() const noexcept {
    return simd_mask{__builtin_convertvector(!data(), simd_mask_impl_t)};
  }

  // Logic and bitwise operators.
  template <int N1>
  friend simd_mask<N1> operator&&(const simd_mask<N1> &,
                                  const simd_mask<N1> &) noexcept;
  template <int N1>
  friend simd_mask<N1> operator||(const simd_mask<N1> &,
                                  const simd_mask<N1> &) noexcept;
  template <int N1>
  friend simd_mask<N1> operator&(const simd_mask<N1> &,
                                 const simd_mask<N1> &) noexcept;
  template <int N1>
  friend simd_mask<N1> operator|(const simd_mask<N1> &,
                                 const simd_mask<N1> &) noexcept;
  template <int N1>
  friend simd_mask<N1> operator^(const simd_mask<N1> &,
                                 const simd_mask<N1> &) noexcept;

  // Comparison operators.
  template <int N1>
  friend simd_mask<N1> operator==(const simd_mask<N1> &,
                                  const simd_mask<N1> &) noexcept;
  template <int N1>
  friend simd_mask<N1> operator!=(const simd_mask<N1> &,
                                  const simd_mask<N1> &) noexcept;

  // Compound assignment operators.
  template <int N1>
  friend simd_mask<N1> &operator&=(simd_mask<N1> &,
                                   const simd_mask<N1> &) noexcept;
  template <int N1>
  friend simd_mask<N1> &operator|=(simd_mask<N1> &,
                                   const simd_mask<N1> &) noexcept;
  template <int N1>
  friend simd_mask<N1> &operator^=(simd_mask<N1> &,
                                   const simd_mask<N1> &) noexcept;

  // TODO
  // template <int N1> friend bool all_of(const simd_mask<N1>&) noexcept;
  // template <int N1> friend bool any_of(const simd_mask<N1>&) noexcept;
  // template <int N1> friend bool none_of(const simd_mask<N1>&) noexcept;
  // template <int N1> friend bool some_of(const simd_mask<N1>&) noexcept;
  // template <int N1> friend int popcount(const simd_mask<N1>&) noexcept;
  // template <int N1> friend int find_first_set(const simd_mask<N1>&);
  // template <int N1> friend int find_last_set(const simd_mask<N1>&);

  // APIs not present in std::experimental:

  // To allow simd_mask<N> m({1,0,0,1,...}).
  template <typename T1> explicit simd_mask(const T1(&&arr)[N]) noexcept {
    for (auto I = 0; I < N; ++I) {
      Val[I] = arr[I]; // implicit conversion from T1 to value_type
    }
  }

  // Load the mask's value from array.
  void copy_from(const value_type (&arr)[N]) {
    simd_mask_impl_t Tmp;
    for (auto I = 0; I < N; ++I) {
      Tmp[I] = arr[I];
    }
    set(Tmp);
  }

  // Store the mask's value to array.
  void copy_to(value_type (&arr)[N]) const {
    for (auto I = 0; I < N; ++I) {
      arr[I] = data()[I];
    }
  }

  // Assignment operator to support simd_mask<N> n = a > b;
  simd_mask &operator=(value_type val) noexcept {
    set(val);
    return *this;
  }

  // TODO FIXME Make this private, add friend accessor to let ESIMD API free
  // functions access the 'Val' field. Mimics simd::data() - the note applies
  // there too.
  simd_mask_impl_t data() const {
#ifndef __SYCL_DEVICE_ONLY__
    return Val;
#else
    return __esimd_vload<value_type, N>(&Val);
#endif // __SYCL_DEVICE_ONLY__
  }

#define __ESIMD_MASK_DEPRECATION_MSG                                           \
  "Use of 'simd' class to represent predicate or mask is deprecated. Use "     \
  "'simd_mask' instead."

  // Implicit conversion constructors for backward compatibility
  // (implemented in esimd.hpp)
  __SYCL_DEPRECATED(__ESIMD_MASK_DEPRECATION_MSG)
  simd_mask(const simd<unsigned short, N> &) noexcept;

  __SYCL_DEPRECATED(__ESIMD_MASK_DEPRECATION_MSG)
  simd_mask(simd<unsigned short, N> &&) noexcept;

  // Assignment operators for backward compatibility (implemented in esimd.hpp)
  __SYCL_DEPRECATED(__ESIMD_MASK_DEPRECATION_MSG)
  simd_mask &operator=(const simd<unsigned short, N> &) noexcept;

  __SYCL_DEPRECATED(__ESIMD_MASK_DEPRECATION_MSG)
  simd_mask &operator=(simd<unsigned short, N> &&) noexcept;

#undef __ESIMD_MASK_DEPRECATION_MSG

private:
  explicit simd_mask(const simd_mask_impl_t &data) noexcept { set(data); }
  explicit simd_mask(simd_mask_impl_t &&data) noexcept { set(data); }

  // Factory function to create simd_mask objects from a result of a clang
  // vector binary operation. rvalue reference version.
  template <typename T1>
  static simd_mask<N> create(detail::mask_impl_t<T1, N> &&v) noexcept {
    return simd_mask<N>{__builtin_convertvector(v, simd_mask_impl_t)};
  }

  // Factory function to create simd_mask objects from a result of a clang
  // vector binary operation. Constant reference version.
  template <typename T1>
  static simd_mask<N> create(const detail::mask_impl_t<T1, N> &v) noexcept {
    return simd_mask<N>{__builtin_convertvector(v, simd_mask_impl_t)};
  }

  template <typename, int> friend class simd;
  template <typename, typename> friend class simd_view;

  void set(const simd_mask_impl_t &v) {
#ifndef __SYCL_DEVICE_ONLY__
    Val = v;
#else
    __esimd_vstore<value_type, N>(&Val, v);
#endif
  }

private:
  simd_mask_impl_t Val;
};

// Alias for backward compatibility.
template <int N> using mask_type_t = simd_mask<N>;

// Logic and bitwise operators.
// Comparison operators.

#define __DEFINE_ESIMD_MASK_BIN_OP(op)                                         \
  template <int N1>                                                            \
  ESIMD_INLINE simd_mask<N1> operator op(const simd_mask<N1> &m1,              \
                                         const simd_mask<N1> &m2) noexcept {   \
    auto Res = m1.data() op m2.data();                                         \
    using T = std::remove_reference_t<decltype(Res[0])>;                       \
    return simd_mask<N1>::template create<T>(std::move(Res));                  \
  }

__DEFINE_ESIMD_MASK_BIN_OP(&&)
__DEFINE_ESIMD_MASK_BIN_OP(||)
__DEFINE_ESIMD_MASK_BIN_OP(&)
__DEFINE_ESIMD_MASK_BIN_OP(|)
__DEFINE_ESIMD_MASK_BIN_OP(^)
__DEFINE_ESIMD_MASK_BIN_OP(==)
__DEFINE_ESIMD_MASK_BIN_OP(!=)

#undef __DEFINE_ESIMD_MASK_BIN_OP

// Compound assignment operators.

#define __DEFINE_ESIMD_MASK_ASSIGN_OP(assign_op, op)                           \
  template <int N1>                                                            \
  ESIMD_INLINE simd_mask<N1> &operator assign_op(                              \
      simd_mask<N1> &m1, const simd_mask<N1> &m2) noexcept {                   \
    m1.set(m1.data() op m2.data());                                            \
    return m1;                                                                 \
  }

__DEFINE_ESIMD_MASK_ASSIGN_OP(&=, &)
__DEFINE_ESIMD_MASK_ASSIGN_OP(|=, |)
__DEFINE_ESIMD_MASK_ASSIGN_OP(^=, ^)

#undef __DEFINE_ESIMD_MASK_ASSIGN_OP

} // namespace esimd
} // namespace experimental
} // namespace intel
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
