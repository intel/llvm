//==------- kernel_bundle_enums.hpp - SYCL kernel_bundle related enums -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {

enum class bundle_state : char { input = 0, object = 1, executable = 2 };

}
} // namespace sycl
