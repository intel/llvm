//==---------- common.hpp ----- Common declarations ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/defines_elementary.hpp> // for __SYCL_ALWAYS_INLINE
#include <sycl/detail/export.hpp>             // for __SYCL_EXPORT
#include <sycl/detail/pi.h>                   // for pi_int32

#include <array>       // for array
#include <cassert>     // for assert
#include <cstddef>     // for size_t
#include <string>      // for allocator, operator+
#include <type_traits> // for enable_if_t
#include <utility>     // for index_sequence, make_i...

// Default signature enables the passing of user code location information to
// public methods as a default argument.
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

#if !defined(NDEBUG) && (_MSC_VER > 1929 || __has_builtin(__builtin_FILE))
#define __CODELOC_FILE_NAME __builtin_FILE()
#else
#define __CODELOC_FILE_NAME nullptr
#endif

#if _MSC_VER > 1929 || __has_builtin(__builtin_FUNCTION)
#define __CODELOC_FUNCTION __builtin_FUNCTION()
#else
#define __CODELOC_FUNCTION nullptr
#endif

#if _MSC_VER > 1929 || __has_builtin(__builtin_LINE)
#define __CODELOC_LINE __builtin_LINE()
#else
#define __CODELOC_LINE 0
#endif

#if _MSC_VER > 1929 || __has_builtin(__builtin_COLUMN)
#define __CODELOC_COLUMN __builtin_COLUMN()
#else
#define __CODELOC_COLUMN 0
#endif

// Data structure that captures the user code location information using the
// builtin capabilities of the compiler
struct code_location {
  static constexpr code_location
  current(const char *fileName = __CODELOC_FILE_NAME,
          const char *funcName = __CODELOC_FUNCTION,
          unsigned long lineNo = __CODELOC_LINE,
          unsigned long columnNo = __CODELOC_COLUMN) noexcept {
    return code_location(fileName, funcName, lineNo, columnNo);
  }

#undef __CODELOC_FILE_NAME
#undef __CODELOC_FUNCTION
#undef __CODELOC_LINE
#undef __CODELOC_COLUMN

  constexpr code_location(const char *file, const char *func, int line,
                          int col) noexcept
      : MFileName(file), MFunctionName(func), MLineNo(line), MColumnNo(col) {}

  constexpr code_location() noexcept
      : MFileName(nullptr), MFunctionName(nullptr), MLineNo(0), MColumnNo(0) {}

  constexpr unsigned long lineNumber() const noexcept { return MLineNo; }
  constexpr unsigned long columnNumber() const noexcept { return MColumnNo; }
  constexpr const char *fileName() const noexcept { return MFileName; }
  constexpr const char *functionName() const noexcept { return MFunctionName; }

private:
  const char *MFileName;
  const char *MFunctionName;
  unsigned long MLineNo;
  unsigned long MColumnNo;
};

/// @brief Data type that manages the code_location information in TLS
/// @details As new SYCL features are added, they all enable the propagation of
/// the code location information where the SYCL API was called by the
/// application layer. In order to facilitate this, the tls_code_loc_t object
/// assists in managing the data in TLS :
///   (1) Populate the information when you at the top level function in the
///   call chain. This is usually the end-user entry point function into SYCL.
///   (2) Remove the information when the object goes out of scope in the top
///   level function.
///
/// Usage:-
///   void bar() {
///     tls_code_loc_t p;
///     // Print the source information of where foo() was called in main()
///     std::cout << p.query().fileName() << ":" << p.query().lineNumber() <<
///     std::endl;
///   }
///   // Will work for arbitrary call chain lengths.
///   void bar1() {bar();}
///
///   // Foo() is equivalent to a SYCL end user entry point such as
///   // queue.memcpy() or queue.copy()
///   void foo(const code_location &loc) {
///     tls_code_loc_t tp(loc);
///     bar1();
///   }
///
///   void main() {
///     foo(const code_location &loc = code_location::current());
///   }
class __SYCL_EXPORT tls_code_loc_t {
public:
  /// @brief Consructor that checks to see if a TLS entry already exists
  /// @details If a previous populated TLS entry exists, this constructor will
  /// capture the informationa and allow you to query the information later.
  tls_code_loc_t();
  /// @brief Iniitializes TLS with CodeLoc if a TLS entry not present
  /// @param CodeLoc The code location information to set up the TLS slot with.
  tls_code_loc_t(const detail::code_location &CodeLoc);
  /// If the code location is set up by this instance, reset it.
  ~tls_code_loc_t();
  /// @brief  Query the information in the TLS slot
  /// @return The code location information saved in the TLS slot. If not TLS
  /// entry has been set up, a default coe location is returned.
  const detail::code_location &query();

private:
  // The flag that is used to determine if the object is in a local scope or in
  // the top level scope.
  bool MLocalScope = true;
};

} // namespace detail
} // namespace _V1
} // namespace sycl

#ifdef __SYCL_DEVICE_ONLY__
// TODO remove this when 'assert' is supported in device code
#define __SYCL_ASSERT(x)
#else
#define __SYCL_ASSERT(x) assert(x)
#endif // #ifdef __SYCL_DEVICE_ONLY__

#define __SYCL_PI_ERROR_REPORT                                                 \
  "Native API failed. " /*__FILE__*/                                           \
  /* TODO: replace __FILE__ to report only relative path*/                     \
  /* ":" __SYCL_STRINGIFY(__LINE__) ": " */                                    \
                          "Native API returns: "

#include <sycl/exception.hpp>

// Helper for enabling empty-base optimizations on MSVC.
// TODO: Remove this when MSVC has this optimization enabled by default.
#ifdef _MSC_VER
#define __SYCL_EBO __declspec(empty_bases)
#else
#define __SYCL_EBO
#endif

namespace sycl {
inline namespace _V1 {
namespace detail {
// Produces N-dimensional object of type T whose all components are initialized
// to given integer value.
template <int N, template <int> class T> struct InitializedVal {
  template <int Val> static T<N> get();
};

// Specialization for a one-dimensional type.
template <template <int> class T> struct InitializedVal<1, T> {
  template <int Val> static T<1> get() { return T<1>{Val}; }
};

// Specialization for a two-dimensional type.
template <template <int> class T> struct InitializedVal<2, T> {
  template <int Val> static T<2> get() { return T<2>{Val, Val}; }
};

// Specialization for a three-dimensional type.
template <template <int> class T> struct InitializedVal<3, T> {
  template <int Val> static T<3> get() { return T<3>{Val, Val, Val}; }
};

/// Helper class for the \c NDLoop.
template <int NDims, int Dim, template <int> class LoopBoundTy, typename FuncTy,
          template <int> class LoopIndexTy>
struct NDLoopIterateImpl {
  NDLoopIterateImpl(const LoopIndexTy<NDims> &LowerBound,
                    const LoopBoundTy<NDims> &Stride,
                    const LoopBoundTy<NDims> &UpperBound, FuncTy f,
                    LoopIndexTy<NDims> &Index) {
    constexpr size_t AdjIdx = NDims - 1 - Dim;
    for (Index[AdjIdx] = LowerBound[AdjIdx]; Index[AdjIdx] < UpperBound[AdjIdx];
         Index[AdjIdx] += Stride[AdjIdx]) {

      NDLoopIterateImpl<NDims, Dim - 1, LoopBoundTy, FuncTy, LoopIndexTy>{
          LowerBound, Stride, UpperBound, f, Index};
    }
  }
};

// Specialization for Dim=0 to terminate recursion
template <int NDims, template <int> class LoopBoundTy, typename FuncTy,
          template <int> class LoopIndexTy>
struct NDLoopIterateImpl<NDims, 0, LoopBoundTy, FuncTy, LoopIndexTy> {
  NDLoopIterateImpl(const LoopIndexTy<NDims> &LowerBound,
                    const LoopBoundTy<NDims> &Stride,
                    const LoopBoundTy<NDims> &UpperBound, FuncTy f,
                    LoopIndexTy<NDims> &Index) {

    constexpr size_t AdjIdx = NDims - 1;
    for (Index[AdjIdx] = LowerBound[AdjIdx]; Index[AdjIdx] < UpperBound[AdjIdx];
         Index[AdjIdx] += Stride[AdjIdx]) {

      f(Index);
    }
  }
};

/// Generates an NDims-dimensional perfect loop nest. The purpose of this class
/// is to better support handling of situations where there must be a loop nest
/// over a multi-dimensional space - it allows to avoid generating unnecessary
/// outer loops like 'for (int z=0; z<1; z++)' in case of 1D and 2D iteration
/// spaces or writing specializations of the algorithms for 1D, 2D and 3D cases.
/// Loop is unrolled in a reverse directions, i.e. ID = 0 is the inner-most one.
template <int NDims> struct NDLoop {
  /// Generates ND loop nest with {0,..0} .. \c UpperBound bounds with unit
  /// stride. Applies \c f at each iteration, passing current index of
  /// \c LoopIndexTy<NDims> type as the parameter.
  template <template <int> class LoopBoundTy, typename FuncTy,
            template <int> class LoopIndexTy = LoopBoundTy>
  static __SYCL_ALWAYS_INLINE void iterate(const LoopBoundTy<NDims> &UpperBound,
                                           FuncTy f) {
    const LoopIndexTy<NDims> LowerBound =
        InitializedVal<NDims, LoopIndexTy>::template get<0>();
    const LoopBoundTy<NDims> Stride =
        InitializedVal<NDims, LoopBoundTy>::template get<1>();
    LoopIndexTy<NDims> Index =
        InitializedVal<NDims, LoopIndexTy>::template get<0>();

    NDLoopIterateImpl<NDims, NDims - 1, LoopBoundTy, FuncTy, LoopIndexTy>{
        LowerBound, Stride, UpperBound, f, Index};
  }

  /// Generates ND loop nest with \c LowerBound .. \c UpperBound bounds and
  /// stride \c Stride. Applies \c f at each iteration, passing current index of
  /// \c LoopIndexTy<NDims> type as the parameter.
  template <template <int> class LoopBoundTy, typename FuncTy,
            template <int> class LoopIndexTy = LoopBoundTy>
  static __SYCL_ALWAYS_INLINE void iterate(const LoopIndexTy<NDims> &LowerBound,
                                           const LoopBoundTy<NDims> &Stride,
                                           const LoopBoundTy<NDims> &UpperBound,
                                           FuncTy f) {
    LoopIndexTy<NDims> Index =
        InitializedVal<NDims, LoopIndexTy>::template get<0>();
    NDLoopIterateImpl<NDims, NDims - 1, LoopBoundTy, FuncTy, LoopIndexTy>{
        LowerBound, Stride, UpperBound, f, Index};
  }
};

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

template <typename T> struct InlineVariableHelper {
  static constexpr T value{};
};

template <typename T> constexpr T InlineVariableHelper<T>::value;

// The function extends or truncates number of dimensions of objects of id
// or ranges classes. When extending the new values are filled with
// DefaultValue, truncation just removes extra values.
template <int NewDim, int DefaultValue, template <int> class T, int OldDim>
static T<NewDim> convertToArrayOfN(T<OldDim> OldObj) {
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

// Helper function for creating an arbitrary sized array with the same value
// repeating.
template <typename T, size_t... Is>
static constexpr std::array<T, sizeof...(Is)>
RepeatValueHelper(const T &Arg, std::index_sequence<Is...>) {
  auto ReturnArg = [&](size_t) { return Arg; };
  return {ReturnArg(Is)...};
}
template <size_t N, typename T>
static constexpr std::array<T, N> RepeatValue(const T &Arg) {
  return RepeatValueHelper(Arg, std::make_index_sequence<N>());
}

// to output exceptions caught in ~destructors
#ifndef NDEBUG
#define __SYCL_REPORT_EXCEPTION_TO_STREAM(str, e)                              \
  {                                                                            \
    std::cerr << str << " " << e.what() << std::endl;                          \
    assert(false);                                                             \
  }
#else
#define __SYCL_REPORT_EXCEPTION_TO_STREAM(str, e)
#endif

} // namespace detail
} // namespace _V1
} // namespace sycl
