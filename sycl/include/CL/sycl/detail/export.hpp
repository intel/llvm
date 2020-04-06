//==---------------- export.hpp - SYCL standard header file ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#ifndef SYCL_DEVICE_ONLY
#ifndef __SYCL_EXPORT
#ifdef _WIN32

// MSVC discourages export of classes, that use STL class in API. This
// results in a warning, treated as compile error. Silence C4251 to workaround.
#pragma warning(disable : 4251)
#pragma warning(disable : 4275)

#define DLL_LOCAL

#if __SYCL_BUILD_SYCL_DLL
#define __SYCL_EXPORT __declspec(dllexport)
#define __SYCL_EXPORT_DEPRECATED(x) __declspec(dllexport, deprecated(x))
#else
#define __SYCL_EXPORT __declspec(dllimport)
#define __SYCL_EXPORT_DEPRECATED(x) __declspec(dllimport, deprecated(x))
#endif
#else

#define DLL_LOCAL __attribute__((visibility("hidden")))

#define __SYCL_EXPORT __attribute__((visibility("default")))
#define __SYCL_EXPORT_DEPRECATED(x)                                            \
  __attribute__((visibility("default"), deprecated(x)))
#endif
#endif
#endif
