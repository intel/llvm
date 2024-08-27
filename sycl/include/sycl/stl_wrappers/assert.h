//==---------------- <assert.h> wrapper around STL--------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Must not be guarded. C++ standard says the macro assert is redefined
// according to the current state of NDEBUG each time that <cassert> is
// included.

// <cassert> and <assert.h> are functionally equivalent
#include <cassert>
