//==---------- common.hpp ----- Common declarations ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/defines.hpp>
#include <sycl/detail/defines_elementary.hpp>
#include <sycl/detail/export.hpp>
#include <sycl/detail/pi.hpp>
#include <sycl/detail/stl_type_traits.hpp>

#include <cstdint>
#include <string>

// Default signature enables the passing of user code location information to
// public methods as a default argument. If the end-user wants to disable the
// code location information, they must compile the code with
// -DDISABLE_SYCL_INSTRUMENTATION_METADATA flag
namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {

// The check for output iterator is commented out as it blocks set_final_data
// with void * argument to be used.
// TODO: Align these checks with the SYCL specification when the behaviour
// with void * is clarified.
template <typename DataT>
using EnableIfOutputPointerT = detail::enable_if_t<
    /*is_output_iterator<DataT>::value &&*/ std::is_pointer<DataT>::value>;

template <typename DataT>
using EnableIfOutputIteratorT = detail::enable_if_t<
    /*is_output_iterator<DataT>::value &&*/ !std::is_pointer<DataT>::value>;

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

// The C++ FE may instrument user calls with code location metadata.
// If it does then that will appear as an extra last argument.
// Having _TWO_ mid-param #ifdefs makes the functions very difficult to read.
// Here we simplify the &CodeLoc declaration to be _CODELOCPARAM(&CodeLoc) and
// _CODELOCARG(&CodeLoc).

#ifndef DISABLE_SYCL_INSTRUMENTATION_METADATA
#define _CODELOCONLYPARAM(a)                                                   \
  const detail::code_location a = detail::code_location::current()
#define _CODELOCPARAM(a)                                                       \
  , const detail::code_location a = detail::code_location::current()
#define _CODELOCPARAMDEF(a) , const detail::code_location a

#define _CODELOCARG(a)
#define _CODELOCFW(a) , a
#else
#define _CODELOCONLYPARAM(a)
#define _CODELOCPARAM(a)

#define _CODELOCARG(a) const detail::code_location a = {}
#define _CODELOCFW(a)
#endif

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
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {

__SYCL_EXPORT const char *stringifyErrorCode(pi_int32 error);

static inline std::string codeToString(pi_int32 code) {
  return std::string(std::to_string(code) + " (" + stringifyErrorCode(code) +
                     ")");
}

} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
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

#ifndef __SYCL_SUPPRESS_PI_ERROR_REPORT
#include <sycl/detail/iostream_proxy.hpp>
// TODO: rename all names with direct use of OCL/OPENCL to be backend agnostic.
#define __SYCL_REPORT_PI_ERR_TO_STREAM(expr)                                   \
  {                                                                            \
    auto code = expr;                                                          \
    if (code != PI_SUCCESS) {                                                  \
      std::cerr << __SYCL_PI_ERROR_REPORT << sycl::detail::codeToString(code)  \
                << std::endl;                                                  \
    }                                                                          \
  }
#endif

#ifndef SYCL_SUPPRESS_EXCEPTIONS
#include <sycl/exception.hpp>
// SYCL 1.2.1 exceptions
#define __SYCL_REPORT_PI_ERR_TO_EXC(expr, exc, str)                            \
  {                                                                            \
    auto code = expr;                                                          \
    if (code != PI_SUCCESS) {                                                  \
      std::string err_str =                                                    \
          str ? "\n" + std::string(str) + "\n" : std::string{};                \
      throw exc(__SYCL_PI_ERROR_REPORT + sycl::detail::codeToString(code) +    \
                    err_str,                                                   \
                code);                                                         \
    }                                                                          \
  }
#define __SYCL_REPORT_PI_ERR_TO_EXC_THROW(code, exc, str)                      \
  __SYCL_REPORT_PI_ERR_TO_EXC(code, exc, str)
#define __SYCL_REPORT_PI_ERR_TO_EXC_BASE(code)                                 \
  __SYCL_REPORT_PI_ERR_TO_EXC(code, sycl::runtime_error, nullptr)
#else
#define __SYCL_REPORT_PI_ERR_TO_EXC_BASE(code)                                 \
  __SYCL_REPORT_PI_ERR_TO_STREAM(code)
#endif
// SYCL 2020 exceptions
#define __SYCL_REPORT_ERR_TO_EXC_VIA_ERRC(expr, errc)                          \
  {                                                                            \
    auto code = expr;                                                          \
    if (code != PI_SUCCESS) {                                                  \
      throw sycl::exception(sycl::make_error_code(errc),                       \
                            __SYCL_PI_ERROR_REPORT +                           \
                                sycl::detail::codeToString(code));             \
    }                                                                          \
  }
#define __SYCL_REPORT_ERR_TO_EXC_THROW_VIA_ERRC(code, errc)                    \
  __SYCL_REPORT_ERR_TO_EXC_VIA_ERRC(code, errc)

#ifdef __SYCL_SUPPRESS_PI_ERROR_REPORT
// SYCL 1.2.1 exceptions
#define __SYCL_CHECK_OCL_CODE(X) (void)(X)
#define __SYCL_CHECK_OCL_CODE_THROW(X, EXC, STR)                               \
  {                                                                            \
    (void)(X);                                                                 \
    (void)(STR);                                                               \
  }
#define __SYCL_CHECK_OCL_CODE_NO_EXC(X) (void)(X)
// SYCL 2020 exceptions
#define __SYCL_CHECK_CODE_THROW_VIA_ERRC(X, ERRC) (void)(X)
#else
// SYCL 1.2.1 exceptions
#define __SYCL_CHECK_OCL_CODE(X) __SYCL_REPORT_PI_ERR_TO_EXC_BASE(X)
#define __SYCL_CHECK_OCL_CODE_THROW(X, EXC, STR)                               \
  __SYCL_REPORT_PI_ERR_TO_EXC_THROW(X, EXC, STR)
#define __SYCL_CHECK_OCL_CODE_NO_EXC(X) __SYCL_REPORT_PI_ERR_TO_STREAM(X)
// SYCL 2020 exceptions
#define __SYCL_CHECK_CODE_THROW_VIA_ERRC(X, ERRC)                              \
  __SYCL_REPORT_ERR_TO_EXC_THROW_VIA_ERRC(X, ERRC)
#endif

// Helper for enabling empty-base optimizations on MSVC.
// TODO: Remove this when MSVC has this optimization enabled by default.
#ifdef _MSC_VER
#define __SYCL_EBO __declspec(empty_bases)
#else
#define __SYCL_EBO
#endif

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {

// Helper function for extracting implementation from SYCL's interface objects.
// Note! This function relies on the fact that all SYCL interface classes
// contain "impl" field that points to implementation object. "impl" field
// should be accessible from this function.
//
// Note that due to a bug in MSVC compilers (including MSVC2019 v19.20), it
// may not recognize the usage of this function in friend member declarations
// if the template parameter name there is not equal to the name used here,
// i.e. 'Obj'. For example, using 'Obj' here and 'T' in such declaration
// would trigger that error in MSVC:
//   template <class T>
//   friend decltype(T::impl) detail::getSyclObjImpl(const T &SyclObject);
template <class Obj> decltype(Obj::impl) getSyclObjImpl(const Obj &SyclObject) {
  assert(SyclObject.impl && "every constructor should create an impl");
  return SyclObject.impl;
}

// Returns the raw pointer to the impl object of given face object. The caller
// must make sure the returned pointer is not captured in a field or otherwise
// stored - i.e. must live only as on-stack value.
template <class T>
typename detail::add_pointer_t<typename decltype(T::impl)::element_type>
getRawSyclObjImpl(const T &SyclObject) {
  return SyclObject.impl.get();
}

// Helper function for creation SYCL interface objects from implementations.
// Note! This function relies on the fact that all SYCL interface classes
// contain "impl" field that points to implementation object. "impl" field
// should be accessible from this function.
template <class T> T createSyclObjFromImpl(decltype(T::impl) ImplObj) {
  return T(ImplObj);
}

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

// Kernel set ID, used to group kernels (represented by OSModule & kernel name
// pairs) into disjoint sets based on the kernel distribution among device
// images.
using KernelSetId = size_t;
// Kernel set ID for kernels contained within the SPIR-V file specified via
// environment.
constexpr KernelSetId SpvFileKSId = 0;
constexpr KernelSetId LastKSId = SpvFileKSId;

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

} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
