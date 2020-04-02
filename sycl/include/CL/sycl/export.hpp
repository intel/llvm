//==---------------- export.hpp - SYCL standard header file ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#ifndef SYCL_API
#ifdef _WIN32

// MSVC discourages export of classes, that use STL class in API. This
// results in a warning, treated as compile error. Silence C4251 to workaround.
#pragma warning(disable : 4251)
#pragma warning(disable : 4275)

#if __SYCL_BUILD_SYCL_DLL
#define SYCL_API __declspec(dllexport)
#define SYCL_API_DEPRECATED(x) __declspec(dllexport, deprecated)
#else
#define SYCL_API __declspec(dllimport)
#define SYCL_API_DEPRECATED(x) __declspec(dllimport, deprecated)
#endif
#else
#define SYCL_API __attribute__((visibility("default")))
#define SYCL_API_DEPRECATED(x)                                                 \
  __attribute__((visibility("default"), deprecated(x)))
#endif
#endif
