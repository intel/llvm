/*
 *
 * Copyright (C) 2022-2023 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 */

#include "ur_lib.h"

namespace ur_lib {

void __attribute__((constructor)) createLibContext() {
    context = new context_t;
}

void __attribute__((destructor)) deleteLibContext() { delete context; }

} // namespace ur_lib
