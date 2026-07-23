//==---------- common.hpp ----- Common declarations ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/assert.hpp>
#include <sycl/detail/export.hpp> // for __SYCL_EXPORT
#include <sycl/detail/nd_loop.hpp>

#include <array>       // for array
#include <cstddef>     // for size_t
#include <type_traits> // for enable_if_t
#include <utility>     // for index_sequence, make_i...

namespace sycl {
inline namespace _V1 {
namespace detail {

// The check for output iterator is commented out as it blocks set_final_data
// with void * argument to be used.
// TODO: Align these checks with the SYCL specification when the behaviour
// with void * is clarified.
template <typename DataT>
using EnableIfOutputPointerT = std::enable_if_t<
    /*is_output_iterator<DataT>::value &&*/ std::is_pointer_v<DataT>>;

template <typename DataT>
using EnableIfOutputIteratorT = std::enable_if_t<
    /*is_output_iterator<DataT>::value &&*/ !std::is_pointer_v<DataT>>;

constexpr size_t getNextPowerOfTwoHelper(size_t Var, size_t Offset) {
  return Offset != 64
             ? getNextPowerOfTwoHelper(Var | (Var >> Offset), Offset * 2)
             : Var;
}

// Returns the smallest power of two not less than Var
constexpr size_t getNextPowerOfTwo(size_t Var) {
  return getNextPowerOfTwoHelper(Var - 1, 1) + 1;
}

// Returns linear index by given index and range
template <int Dims, template <int> class T, template <int> class U>
size_t getLinearIndex(const T<Dims> &Index, const U<Dims> &Range) {
  size_t LinearIndex = 0;
  for (int I = 0; I < Dims; ++I)
    LinearIndex = LinearIndex * Range[I] + Index[I];
  return LinearIndex;
}

// The function extends or truncates number of dimensions of objects of id
// or ranges classes. When extending the new values are filled with
// DefaultValue, truncation just removes extra values.
template <int NewDim, int DefaultValue, template <int> class T, int OldDim>
T<NewDim> convertToArrayOfN(T<OldDim> OldObj) {
  T<NewDim> NewObj = detail::InitializedVal<NewDim, T>::template get<0>();
  const int CopyDims = NewDim > OldDim ? OldDim : NewDim;
  for (int I = 0; I < CopyDims; ++I)
    NewObj[I] = OldObj[I];
  for (int I = CopyDims; I < NewDim; ++I)
    NewObj[I] = DefaultValue;
  return NewObj;
}

// Helper function for concatenating two std::array.
template <typename T, std::size_t... Is1, std::size_t... Is2>
constexpr std::array<T, sizeof...(Is1) + sizeof...(Is2)>
ConcatArrays(const std::array<T, sizeof...(Is1)> &A1,
             const std::array<T, sizeof...(Is2)> &A2,
             std::index_sequence<Is1...>, std::index_sequence<Is2...>) {
  return {A1[Is1]..., A2[Is2]...};
}
template <typename T, std::size_t N1, std::size_t N2>
constexpr std::array<T, N1 + N2> ConcatArrays(const std::array<T, N1> &A1,
                                              const std::array<T, N2> &A2) {
  return ConcatArrays(A1, A2, std::make_index_sequence<N1>(),
                      std::make_index_sequence<N2>());
}

// Utility for creating an std::array from the results of flattening the
// arguments using a flattening functor.
template <typename DataT, template <typename, typename> typename FlattenF,
          typename... ArgTN>
struct ArrayCreator;
template <typename DataT, template <typename, typename> typename FlattenF,
          typename ArgT, typename... ArgTN>
struct ArrayCreator<DataT, FlattenF, ArgT, ArgTN...> {
  static constexpr auto Create(const ArgT &Arg, const ArgTN &...Args) {
    auto ImmArray = FlattenF<DataT, ArgT>()(Arg);
    // Due to a bug in MSVC narrowing size_t to a bool in an if constexpr causes
    // warnings. To avoid this we add the comparison to 0.
    if constexpr (sizeof...(Args) > 0)
      return ConcatArrays(
          ImmArray, ArrayCreator<DataT, FlattenF, ArgTN...>::Create(Args...));
    else
      return ImmArray;
  }
};
template <typename DataT, template <typename, typename> typename FlattenF>
struct ArrayCreator<DataT, FlattenF> {
  static constexpr auto Create() { return std::array<DataT, 0>{}; }
};

// to output exceptions caught in ~destructors.
// The symbol is always declared/exported regardless of the consumer's NDEBUG
// state so a debug TU never links against a release-built library and gets an
// undefined reference; only the macro that *calls* it is gated.
__SYCL_EXPORT void reportExceptionToStream(const char *Prefix,
                                           const char *What);
#ifndef NDEBUG
#define __SYCL_REPORT_EXCEPTION_TO_STREAM(str, e)                              \
  do {                                                                         \
    ::sycl::_V1::detail::reportExceptionToStream(str, e.what());               \
    assert(false);                                                             \
  } while (0)
#else
#define __SYCL_REPORT_EXCEPTION_TO_STREAM(str, e) (void)e;
#endif

// Tag to help create CTAD definition to avoid ctad-maybe-unsupported warning
// in GCC when relying on default deductions on non-template ctors in template
// classes.
struct AllowCTADTag;

} // namespace detail
} // namespace _V1
} // namespace sycl
