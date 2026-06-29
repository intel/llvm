/*
 *
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM
 * Exceptions. See https://llvm.org/LICENSE.txt for license information.
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */
#ifndef UR_REF_COUNT_HPP
#define UR_REF_COUNT_HPP 1

#include <atomic>
#include <cstdint>

namespace ur {

class RefCount {
public:
  RefCount(uint32_t count = 1) : Count(count) {}
  RefCount(const RefCount &) = delete;
  RefCount &operator=(const RefCount &) = delete;

  uint32_t getCount() const noexcept { return Count.load(); }
  uint32_t retain() { return ++Count; }
  bool release() { return --Count == 0; }
  void reset(uint32_t value = 1) { Count = value; }

private:
  std::atomic_uint32_t Count;
};

} // namespace ur

#endif // UR_REF_COUNT_HPP
