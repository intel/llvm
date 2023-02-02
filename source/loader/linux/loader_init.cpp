/*
 *
 * Copyright (C) 2022-2023 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 */

#include "ur_loader.hpp"

namespace loader {

void __attribute__((constructor)) createLoaderContext() {
    context = new context_t;
}

void __attribute__((destructor)) deleteLoaderContext() { delete context; }

} // namespace loader
