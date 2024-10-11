// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <map>
#include <string>

// Map used by L0 adapter to count the number of calls to each L0 function
// Lifetime is managed by the adapter, this variable is defined here
// only so that we can read it from the tests.
__attribute__((visibility("default"))) std::map<std::string, int> *ZeCallCount =
    nullptr;
