// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef UR_UNIT_TEST_HELPERS_H
#define UR_UNIT_TEST_HELPERS_H 1

#if defined(_WIN32)
#define setenv(name, value, overwrite) _putenv_s(name, value)
#define unsetenv(name) _putenv_s(name, "")
#endif

#endif /* UR_UNIT_TEST_HELPERS_H */
