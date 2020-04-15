//==-------------- backend_types.hpp - SYCL backend types ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

namespace cl {
namespace sycl {

enum class backend { host, opencl, cuda };

template <backend name, typename SYCLObjectT> struct interop;

} // namespace sycl
} // namespace cl