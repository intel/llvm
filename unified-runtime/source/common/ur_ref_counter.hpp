/*
 *
 * Copyright (C) 2025 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */
#ifndef UR_REF_COUNTER_HPP
#define UR_REF_COUNTER_HPP 1

#include <cstdint>
#include <atomic>

class UR_ReferenceCounter {
public:
  uint32_t getCount() const noexcept { return Count.load(); }
  uint32_t increment() { return ++Count; }
  uint32_t decrement() { return --Count; }
  void reset() { Count = 1; }

private:
  std::atomic_uint32_t Count{1};
};

#endif // UR_REF_COUNTER_HPP
