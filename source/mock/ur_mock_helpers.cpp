/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT SPDX-License-Identifier: Apache-2.0 WITH
 * LLVM-exception
 *
 * @file ur_mock_helpers.cpp
 *
 */

#include "ur_mock_helpers.hpp"

namespace mock {
static callbacks_t callbacks = {};

callbacks_t &getCallbacks() { return callbacks; }

} // namespace mock
