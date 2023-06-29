/*
 *
 * Copyright (C) 2022-2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "ur_lib.hpp"

namespace ur_lib {

void __attribute__((constructor)) createLibContext() {
    context = new context_t;
}

void __attribute__((destructor)) deleteLibContext() { delete context; }

} // namespace ur_lib
