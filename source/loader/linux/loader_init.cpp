/*
 *
 * Copyright (C) 2022-2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "ur_loader.hpp"

namespace ur_loader {

void __attribute__((constructor)) createLoaderContext() {
    context = new context_t;
}

void __attribute__((destructor)) deleteLoaderContext() { delete context; }

} // namespace ur_loader
