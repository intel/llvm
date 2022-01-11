//==---------- common.hpp ----- Common declarations ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/cl.h>
#include <CL/sycl/detail/defines.hpp>
#include <CL/sycl/detail/defines_elementary.hpp>
#include <CL/sycl/detail/export.hpp>
#include <CL/sycl/detail/stl_type_traits.hpp>

#include <cstdint>
#include <string>

// Default signature enables the passing of user code location information to
// public methods as a default argument. If the end-user wants to disable the
// code location information, they must compile the code with
// -DDISABLE_SYCL_INSTRUMENTATION_METADATA flag
__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

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
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

__SYCL_EXPORT const char *stringifyErrorCode(cl_int error);

static inline std::string codeToString(cl_int code) {
  return std::string(std::to_string(code) + " (" + stringifyErrorCode(code) +
                     ")");
}

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

#ifdef __SYCL_DEVICE_ONLY__
// TODO remove this when 'assert' is supported in device code
#define __SYCL_ASSERT(x)
#else
#define __SYCL_ASSERT(x) assert(x)
#endif // #ifdef __SYCL_DEVICE_ONLY__

#define __SYCL_OCL_ERROR_REPORT                                                \
  "Native API failed. " /*__FILE__*/                                           \
  /* TODO: replace __FILE__ to report only relative path*/                     \
  /* ":" __SYCL_STRINGIFY(__LINE__) ": " */                                    \
                          "Native API returns: "

#ifndef __SYCL_SUPPRESS_OCL_ERROR_REPORT
#include <iostream>
// TODO: rename all names with direct use of OCL/OPENCL to be backend agnostic.
#define __SYCL_REPORT_OCL_ERR_TO_STREAM(expr)                                  \
  {                                                                            \
    auto code = expr;                                                          \
    if (code != CL_SUCCESS) {                                                  \
      std::cerr << __SYCL_OCL_ERROR_REPORT                                     \
                << cl::sycl::detail::codeToString(code) << std::endl;          \
    }                                                                          \
  }
#endif

#ifndef SYCL_SUPPRESS_EXCEPTIONS
#include <CL/sycl/exception.hpp>
// SYCL 1.2.1 exceptions
#define __SYCL_REPORT_OCL_ERR_TO_EXC(expr, exc)                                \
  {                                                                            \
    auto code = expr;                                                          \
    if (code != CL_SUCCESS) {                                                  \
      throw exc(__SYCL_OCL_ERROR_REPORT +                                      \
                    cl::sycl::detail::codeToString(code),                      \
                code);                                                         \
    }                                                                          \
  }
#define __SYCL_REPORT_OCL_ERR_TO_EXC_THROW(code, exc)                          \
  __SYCL_REPORT_OCL_ERR_TO_EXC(code, exc)
#define __SYCL_REPORT_OCL_ERR_TO_EXC_BASE(code)                                \
  __SYCL_REPORT_OCL_ERR_TO_EXC(code, cl::sycl::runtime_error)
#else
#define __SYCL_REPORT_OCL_ERR_TO_EXC_BASE(code)                                \
  __SYCL_REPORT_OCL_ERR_TO_STREAM(code)
#endif
// SYCL 2020 exceptions
#define __SYCL_REPORT_ERR_TO_EXC_VIA_ERRC(expr, errc)                          \
  {                                                                            \
    auto code = expr;                                                          \
    if (code != CL_SUCCESS) {                                                  \
      throw sycl::exception(sycl::make_error_code(errc),                       \
                            __SYCL_OCL_ERROR_REPORT +                          \
                                cl::sycl::detail::codeToString(code));         \
    }                                                                          \
  }
#define __SYCL_REPORT_ERR_TO_EXC_THROW_VIA_ERRC(code, errc)                    \
  __SYCL_REPORT_ERR_TO_EXC_VIA_ERRC(code, errc)

#ifdef __SYCL_SUPPRESS_OCL_ERROR_REPORT
// SYCL 1.2.1 exceptions
#define __SYCL_CHECK_OCL_CODE(X) (void)(X)
#define __SYCL_CHECK_OCL_CODE_THROW(X, EXC) (void)(X)
#define __SYCL_CHECK_OCL_CODE_NO_EXC(X) (void)(X)
// SYCL 2020 exceptions
#define __SYCL_CHECK_CODE_THROW_VIA_ERRC(X, ERRC) (void)(X)
#else
// SYCL 1.2.1 exceptions
#define __SYCL_CHECK_OCL_CODE(X) __SYCL_REPORT_OCL_ERR_TO_EXC_BASE(X)
#define __SYCL_CHECK_OCL_CODE_THROW(X, EXC)                                    \
  __SYCL_REPORT_OCL_ERR_TO_EXC_THROW(X, EXC)
#define __SYCL_CHECK_OCL_CODE_NO_EXC(X) __SYCL_REPORT_OCL_ERR_TO_STREAM(X)
// SYCL 2020 exceptions
#define __SYCL_CHECK_CODE_THROW_VIA_ERRC(X, ERRC)                              \
  __SYCL_REPORT_ERR_TO_EXC_THROW_VIA_ERRC(X, ERRC)
#endif

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
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
template <int NDIMS, int DIM, template <int> class LoopBoundTy, typename FuncTy,
          template <int> class LoopIndexTy>
struct NDLoopIterateImpl {
  NDLoopIterateImpl(const LoopIndexTy<NDIMS> &LowerBound,
                    const LoopBoundTy<NDIMS> &Stride,
                    const LoopBoundTy<NDIMS> &UpperBound, FuncTy f,
                    LoopIndexTy<NDIMS> &Index) {
    constexpr size_t AdjIdx = NDIMS - 1 - DIM;
    for (Index[AdjIdx] = LowerBound[AdjIdx]; Index[AdjIdx] < UpperBound[AdjIdx];
         Index[AdjIdx] += Stride[AdjIdx]) {

      NDLoopIterateImpl<NDIMS, DIM - 1, LoopBoundTy, FuncTy, LoopIndexTy>{
          LowerBound, Stride, UpperBound, f, Index};
    }
  }
};

// Specialization for DIM=0 to terminate recursion
template <int NDIMS, template <int> class LoopBoundTy, typename FuncTy,
          template <int> class LoopIndexTy>
struct NDLoopIterateImpl<NDIMS, 0, LoopBoundTy, FuncTy, LoopIndexTy> {
  NDLoopIterateImpl(const LoopIndexTy<NDIMS> &LowerBound,
                    const LoopBoundTy<NDIMS> &Stride,
                    const LoopBoundTy<NDIMS> &UpperBound, FuncTy f,
                    LoopIndexTy<NDIMS> &Index) {

    constexpr size_t AdjIdx = NDIMS - 1;
    for (Index[AdjIdx] = LowerBound[AdjIdx]; Index[AdjIdx] < UpperBound[AdjIdx];
         Index[AdjIdx] += Stride[AdjIdx]) {

      f(Index);
    }
  }
};

/// Generates an NDIMS-dimensional perfect loop nest. The purpose of this class
/// is to better support handling of situations where there must be a loop nest
/// over a multi-dimensional space - it allows to avoid generating unnecessary
/// outer loops like 'for (int z=0; z<1; z++)' in case of 1D and 2D iteration
/// spaces or writing specializations of the algorithms for 1D, 2D and 3D cases.
/// Loop is unrolled in a reverse directions, i.e. ID = 0 is the inner-most one.
template <int NDIMS> struct NDLoop {
  /// Generates ND loop nest with {0,..0} .. \c UpperBound bounds with unit
  /// stride. Applies \c f at each iteration, passing current index of
  /// \c LoopIndexTy<NDIMS> type as the parameter.
  template <template <int> class LoopBoundTy, typename FuncTy,
            template <int> class LoopIndexTy = LoopBoundTy>
  static __SYCL_ALWAYS_INLINE void iterate(const LoopBoundTy<NDIMS> &UpperBound,
                                           FuncTy f) {
    const LoopIndexTy<NDIMS> LowerBound =
        InitializedVal<NDIMS, LoopIndexTy>::template get<0>();
    const LoopBoundTy<NDIMS> Stride =
        InitializedVal<NDIMS, LoopBoundTy>::template get<1>();
    LoopIndexTy<NDIMS> Index =
        InitializedVal<NDIMS, LoopIndexTy>::template get<0>();

    NDLoopIterateImpl<NDIMS, NDIMS - 1, LoopBoundTy, FuncTy, LoopIndexTy>{
        LowerBound, Stride, UpperBound, f, Index};
  }

  /// Generates ND loop nest with \c LowerBound .. \c UpperBound bounds and
  /// stride \c Stride. Applies \c f at each iteration, passing current index of
  /// \c LoopIndexTy<NDIMS> type as the parameter.
  template <template <int> class LoopBoundTy, typename FuncTy,
            template <int> class LoopIndexTy = LoopBoundTy>
  static __SYCL_ALWAYS_INLINE void iterate(const LoopIndexTy<NDIMS> &LowerBound,
                                           const LoopBoundTy<NDIMS> &Stride,
                                           const LoopBoundTy<NDIMS> &UpperBound,
                                           FuncTy f) {
    LoopIndexTy<NDIMS> Index =
        InitializedVal<NDIMS, LoopIndexTy>::template get<0>();
    NDLoopIterateImpl<NDIMS, NDIMS - 1, LoopBoundTy, FuncTy, LoopIndexTy>{
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
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
