
/*
 *
 * Copyright (C) 2025 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file sanitizer_common_ur.hpp
 *
 * This file contains the common functions and data structures related to UR.
 *
 */

namespace ur_sanitizer_layer {
void PrintUrBuildLog(ur_program_handle_t hProgram,
                     ur_device_handle_t *phDevices, size_t numDevices);

} // namespace ur_sanitizer_layer
