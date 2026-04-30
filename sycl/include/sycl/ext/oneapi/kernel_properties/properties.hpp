//==------- properties.hpp - compatibility forwarding header --------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

// Compatibility shim for the legacy kernel_properties/* include path.
// Delete this header when support for that include path is removed.
#include <sycl/ext/oneapi/kernel_properties.hpp>
