// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Level Zero v2 adapter regression tests for discrete-buffer host migration
// (see urEnqueueMemBufferMultiDeviceMigration.cpp).

// RUN: %with-v2 ./discrete_buffer_host_migration-test
// REQUIRES: v2

#include "../../../conformance/enqueue/urEnqueueMemBufferMultiDeviceMigration.cpp"
