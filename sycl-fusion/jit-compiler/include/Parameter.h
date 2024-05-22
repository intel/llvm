//==--- Parameter.h - JIT compiler representations for SYCL kernel params --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SYCL_FUSION_JIT_COMPILER_PARAMETER_H
#define SYCL_FUSION_JIT_COMPILER_PARAMETER_H

#include "DynArray.h"

#include <algorithm>
#include <cstdint>

namespace jit_compiler {
///
/// Parameters are identified by the index of the defining kernel
/// in the list of input kernels and the parameter's index in that
/// kernel's parameter list.
struct Parameter {
  unsigned KernelIdx;
  unsigned ParamIdx;

  ///
  /// Compares two instances of Parameter
  friend bool operator==(const Parameter &LHS, const Parameter &RHS) noexcept {
    return LHS.KernelIdx == RHS.KernelIdx && LHS.ParamIdx == RHS.ParamIdx;
  }

  ///
  /// Compares two instances of Parameter
  friend bool operator!=(const Parameter &LHS, const Parameter &RHS) noexcept {
    return !(LHS == RHS);
  }
};

///
/// Express that two parameters, identified by kernel and parameter
/// index have identical value.
struct ParameterIdentity {
  Parameter LHS;
  Parameter RHS;

  ///
  /// Compares two instances of ParameterIdentity
  friend bool operator==(const ParameterIdentity &LHS,
                         const ParameterIdentity &RHS) noexcept {
    return (LHS.LHS == RHS.LHS && LHS.RHS == RHS.RHS) ||
           (LHS.LHS == RHS.RHS && LHS.RHS == RHS.LHS);
  }

  ///
  /// Compares two instances of ParameterIdentity
  friend bool operator!=(const ParameterIdentity &LHS,
                         const ParameterIdentity &RHS) noexcept {
    return !(LHS == RHS);
  }
};

///
/// Express how a parameter can be lowered using promotion to local or global
/// memory.
///
/// 1:1 correspondence with the enum in include/SYCL/common.h (SYCL runtime)
enum class Internalization : unsigned {
  None = 0, /// Not used. Introduced for symmetry with the original enum
  Local = 1,
  Private = 2,
};

///
/// Express that a parameter can be internalized by local or private promotion
/// or that it cannot be internalized at all.
struct ParameterInternalization {
  Parameter Param;
  Internalization Intern;
  std::size_t LocalSize;
  std::size_t ElemSize;
  ParameterInternalization() = default;
  ParameterInternalization(const Parameter &Param, Internalization Intern,
                           std::size_t LocalSize, std::size_t ElemSize)
      : Param{Param}, Intern{Intern}, LocalSize{LocalSize}, ElemSize(ElemSize) {
  }

  friend bool operator==(const ParameterInternalization &LHS,
                         const ParameterInternalization &RHS) noexcept {
    return LHS.LocalSize == RHS.LocalSize && LHS.ElemSize == RHS.ElemSize &&
           LHS.Intern == RHS.Intern && LHS.Param == RHS.Param;
  }

  friend bool operator!=(const ParameterInternalization &LHS,
                         const ParameterInternalization &RHS) noexcept {
    return !(LHS == RHS);
  }
};

///
/// Express that a parameter is a scalar or aggregate whose value is known at
/// JIT-compilation time.
/// Client of the API owns the data held by `ValPtr`.
struct JITConstant {
  Parameter Param;
  DynArray<char> Value;
  JITConstant() = default;
  JITConstant(const Parameter &Parameter, void *Ptr, size_t Size)
      : Param{Parameter}, Value{Size} {
    auto *CPtr = reinterpret_cast<const char *>(Ptr);
    std::copy(CPtr, CPtr + Size, Value.begin());
  }

  friend bool operator==(const JITConstant &LHS,
                         const JITConstant &RHS) noexcept {

    return LHS.Param == RHS.Param && LHS.Value == RHS.Value;
  }

  friend bool operator!=(const JITConstant &LHS,
                         const JITConstant &RHS) noexcept {
    return !(LHS == RHS);
  }
};
} // namespace jit_compiler

#endif // SYCL_FUSION_JIT_COMPILER_PARAMETER_H
