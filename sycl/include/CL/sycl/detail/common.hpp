//==---------- common.hpp ----- Common declarations ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

// Suppress a compiler warning about undefined CL_TARGET_OPENCL_VERSION
// Khronos ICD supports only latest OpenCL version
#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <CL/cl_ext_intel.h>
#include <string>
#include <type_traits>

#define STRINGIFY_LINE_HELP(s) #s
#define STRINGIFY_LINE(s) STRINGIFY_LINE_HELP(s)

const char *stringifyErrorCode(cl_int error);

static inline std::string codeToString(cl_int code){
  return std::string(std::to_string(code) + " (" +
         stringifyErrorCode(code) + ")");
}

#ifdef __SYCL_DEVICE_ONLY__
// TODO remove this when 'assert' is supported in device code
#define __SYCL_ASSERT(x)
#else
#define __SYCL_ASSERT(x) assert(x)
#endif // #ifdef __SYCL_DEVICE_ONLY__

#define OCL_ERROR_REPORT                                                       \
  "OpenCL API failed. " /*__FILE__*/                                           \
  /* TODO: replace __FILE__ to report only relative path*/                     \
  /* ":" STRINGIFY_LINE(__LINE__) ": " */                                      \
                               "OpenCL API returns: "

#ifndef SYCL_SUPPRESS_OCL_ERROR_REPORT
#include <iostream>
#define REPORT_OCL_ERR_TO_STREAM(expr)                                         \
{                                                                              \
  auto code = expr;                                                            \
  if (code != CL_SUCCESS) {                                                    \
    std::cerr << OCL_ERROR_REPORT << codeToString(code) << std::endl;          \
  }                                                                            \
}
#endif

#ifndef SYCL_SUPPRESS_EXCEPTIONS
#include <CL/sycl/exception.hpp>

#define REPORT_OCL_ERR_TO_EXC(expr, exc)                                       \
{                                                                              \
  auto code = expr;                                                            \
  if (code != CL_SUCCESS) {                                                    \
    throw exc(OCL_ERROR_REPORT + codeToString(code), code);                    \
  }                                                                            \
}
#define REPORT_OCL_ERR_TO_EXC_THROW(code, exc) REPORT_OCL_ERR_TO_EXC(code, exc)
#define REPORT_OCL_ERR_TO_EXC_BASE(code)                                       \
  REPORT_OCL_ERR_TO_EXC(code, cl::sycl::runtime_error)
#else
#define REPORT_OCL_ERR_TO_EXC_BASE(code) REPORT_OCL_ERR_TO_STREAM(code)
#endif

#ifdef SYCL_SUPPRESS_OCL_ERROR_REPORT
#define CHECK_OCL_CODE(X) (void)(X)
#define CHECK_OCL_CODE_THROW(X, EXC) (void)(X)
#define CHECK_OCL_CODE_NO_EXC(X) (void)(X)
#else
#define CHECK_OCL_CODE(X) REPORT_OCL_ERR_TO_EXC_BASE(X)
#define CHECK_OCL_CODE_THROW(X, EXC) REPORT_OCL_ERR_TO_EXC_THROW(X, EXC)
#define CHECK_OCL_CODE_NO_EXC(X) REPORT_OCL_ERR_TO_STREAM(X)
#endif

#ifndef __has_attribute
#define __has_attribute(x) 0
#endif

#if __has_attribute(always_inline)
#define ALWAYS_INLINE __attribute__((always_inline))
#else
#define ALWAYS_INLINE
#endif

#ifndef SYCL_EXTERNAL
#define SYCL_EXTERNAL
#endif

namespace cl {
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
typename std::add_pointer<typename decltype(T::impl)::element_type>::type
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

    for (Index[DIM] = LowerBound[DIM]; Index[DIM] < UpperBound[DIM];
         Index[DIM] += Stride[DIM]) {

      NDLoopIterateImpl<NDIMS, DIM - 1, LoopBoundTy, FuncTy, LoopIndexTy>{
          LowerBound, Stride, UpperBound, f, Index};
    }
  }
};

// spcialization for DIM=0 to terminate recursion
template <int NDIMS, template <int> class LoopBoundTy, typename FuncTy,
          template <int> class LoopIndexTy>
struct NDLoopIterateImpl<NDIMS, 0, LoopBoundTy, FuncTy, LoopIndexTy> {
  NDLoopIterateImpl(const LoopIndexTy<NDIMS> &LowerBound,
                    const LoopBoundTy<NDIMS> &Stride,
                    const LoopBoundTy<NDIMS> &UpperBound, FuncTy f,
                    LoopIndexTy<NDIMS> &Index) {

    for (Index[0] = LowerBound[0]; Index[0] < UpperBound[0];
         Index[0] += Stride[0]) {

      f(Index);
    }
  }
};

/// Generates an NDIMS-dimensional perfect loop nest. The purpose of this class
/// is to better support handling of situations where there must be a loop nest
/// over a multi-dimensional space - it allows to avoid generating unnecessary
/// outer loops like 'for (int z=0; z<1; z++)' in case of 1D and 2D iteration
/// spaces or writing specializations of the algorithms for 1D, 2D and 3D cases.
template <int NDIMS> struct NDLoop {
  /// Generates ND loop nest with {0,..0} .. \c UpperBound bounds with unit
  /// stride. Applies \c f at each iteration, passing current index of
  /// \c LoopIndexTy<NDIMS> type as the parameter.
  template <template <int> class LoopBoundTy, typename FuncTy,
            template <int> class LoopIndexTy = LoopBoundTy>
  static ALWAYS_INLINE void iterate(const LoopBoundTy<NDIMS> &UpperBound,
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
  static ALWAYS_INLINE void iterate(const LoopIndexTy<NDIMS> &LowerBound,
                                    const LoopBoundTy<NDIMS> &Stride,
                                    const LoopBoundTy<NDIMS> &UpperBound,
                                    FuncTy f) {
    LoopIndexTy<NDIMS> Index; // initialized down the call stack
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

} // namespace detail
} // namespace sycl
} // namespace cl
