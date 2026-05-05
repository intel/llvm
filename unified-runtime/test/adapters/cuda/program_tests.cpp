// Copyright (C) 2026 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "uur/fixtures.h"
#include "uur/raii.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

using cudaProgramTest = uur::urQueueTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE(cudaProgramTest);

namespace {

const char *kPtxSource = "\n\
.version 3.2\n\
.target sm_20\n\
.address_size 64\n\
.visible .entry _Z8myKernelPi(\n\
\t.param .u64 _Z8myKernelPi_param_0\n\
)\n\
{\n\
\tret;\n\
}\n\
";

} // namespace

TEST_P(cudaProgramTest, CreateWithBinaryDeepCopiesInputBuffer) {
  std::vector<uint8_t> original(reinterpret_cast<const uint8_t *>(kPtxSource),
                                reinterpret_cast<const uint8_t *>(kPtxSource) +
                                    std::strlen(kPtxSource));
  std::vector<uint8_t> source = original;

  size_t binary_size = source.size();
  const uint8_t *binary_data = source.data();

  uur::raii::Program program = nullptr;
  ASSERT_SUCCESS(urProgramCreateWithBinary(
      context, 1, &device, &binary_size, &binary_data, nullptr, program.ptr()));

  // If urProgramCreateWithBinary keeps a borrowed pointer, this overwrite would
  // corrupt the program binary view.
  std::fill(source.begin(), source.end(), 0);

  size_t returned_binary_size = 0;
  ASSERT_SUCCESS(urProgramGetInfo(program, UR_PROGRAM_INFO_BINARY_SIZES,
                                  sizeof(returned_binary_size),
                                  &returned_binary_size, nullptr));
  ASSERT_EQ(returned_binary_size, original.size());

  std::vector<uint8_t> returned_binary(returned_binary_size);
  uint8_t *returned_binary_ptr = returned_binary.data();
  ASSERT_SUCCESS(urProgramGetInfo(program, UR_PROGRAM_INFO_BINARIES,
                                  sizeof(returned_binary_ptr),
                                  &returned_binary_ptr, nullptr));

  EXPECT_EQ(returned_binary, original);
}

TEST_P(cudaProgramTest, LinkProgramOwnsBinaryAfterLinkStateDestroyed) {
  std::vector<uint8_t> source(reinterpret_cast<const uint8_t *>(kPtxSource),
                              reinterpret_cast<const uint8_t *>(kPtxSource) +
                                  std::strlen(kPtxSource));

  size_t binary_size = source.size();
  const uint8_t *binary_data = source.data();

  uur::raii::Program input_program = nullptr;
  ASSERT_SUCCESS(urProgramCreateWithBinary(context, 1, &device, &binary_size,
                                           &binary_data, nullptr,
                                           input_program.ptr()));

  uur::raii::Program linked_program = nullptr;
  ur_program_handle_t input_program_handle = input_program.get();
  ASSERT_SUCCESS(urProgramLink(context, 1, &input_program_handle, nullptr,
                               linked_program.ptr()));

  size_t linked_binary_size = 0;
  ASSERT_SUCCESS(urProgramGetInfo(linked_program, UR_PROGRAM_INFO_BINARY_SIZES,
                                  sizeof(linked_binary_size),
                                  &linked_binary_size, nullptr));
  ASSERT_GT(linked_binary_size, 0u);

  std::vector<uint8_t> linked_binary(linked_binary_size);
  uint8_t *linked_binary_ptr = linked_binary.data();
  ASSERT_SUCCESS(urProgramGetInfo(linked_program, UR_PROGRAM_INFO_BINARIES,
                                  sizeof(linked_binary_ptr), &linked_binary_ptr,
                                  nullptr));
  EXPECT_TRUE(std::any_of(linked_binary.begin(), linked_binary.end(),
                          [](uint8_t byte) { return byte != 0; }));

  // Verify the linked program remains usable after binary retrieval.
  uur::raii::Kernel kernel = nullptr;
  ASSERT_SUCCESS(urKernelCreate(linked_program, "_Z8myKernelPi", kernel.ptr()));
}
