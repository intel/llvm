// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#ifndef UR_UNIT_TEST_HELPERS_H
#define UR_UNIT_TEST_HELPERS_H 1

#if defined(_WIN32)
#define setenv(name, value, overwrite) _putenv_s(name, value)
#define unsetenv(name) _putenv_s(name, "")
#endif

#endif /* UR_UNIT_TEST_HELPERS_H */
