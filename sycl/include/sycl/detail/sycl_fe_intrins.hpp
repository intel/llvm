//==------ sycl_fe_intrins.hpp --- SYCL Device Compiler's FE intrinsics ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// C++ intrinsics recognized by the SYCL device compiler frontend
//===----------------------------------------------------------------------===//

#pragma once

#include <cstddef>
#include <cstdint>

#ifdef __SYCL_DEVICE_ONLY__

// Get the value of the specialization constant with given name.
// Post-link tool traces the ID to a string literal it points to and assigns
// integer ID.
template <typename T>
SYCL_EXTERNAL T __sycl_getScalarSpecConstantValue(const char *ID);

template <typename T>
SYCL_EXTERNAL T __sycl_getCompositeSpecConstantValue(const char *ID);

// The intrinsics below are used to implement support SYCL2020 specialization
// constants. SYCL2020 version requires more parameters compared to the initial
// version.

// Get the value of the specialization constant with given symbolic ID.
// `SymbolicID` is a unique string ID of a specialization constant.
// `DefaultValue` contains a pointer to a global variable with the initializer,
// which should be used as the default value of the specialization constants.
// `RTBuffer` is a pointer to a runtime buffer, which holds values of all
// specialization constant and should be used if native specialization constants
// are not available.
template <typename T>
SYCL_EXTERNAL T __sycl_getScalar2020SpecConstantValue(const char *SymbolicID,
                                                      const void *DefaultValue,
                                                      const void *RTBuffer);

template <typename T>
SYCL_EXTERNAL T __sycl_getComposite2020SpecConstantValue(
    const char *SymbolicID, const void *DefaultValue, const void *RTBuffer);

// Request a fixed-size allocation in local address space at kernel scope.
extern "C" SYCL_EXTERNAL __attribute__((opencl_local)) std::uint8_t *
__sycl_allocateLocalMemory(std::size_t Size, std::size_t Alignment);

#endif
