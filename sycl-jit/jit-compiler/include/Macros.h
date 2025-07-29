//===- Macros.h - Common definitions for SYCL-JIT entrypoints -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#ifdef _WIN32
#define JIT_EXPORT_SYMBOL extern "C" __declspec(dllexport)
#else
#define JIT_EXPORT_SYMBOL extern "C"
#endif

#ifdef __clang__
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
#endif // __clang__

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4190)
#endif // _MSC_VER
