//===---------- usm_allocator.hpp - Allocator for USM memory --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef USM_ALLOCATOR
#define USM_ALLOCATOR

#include <memory>

// USM system memory allocation/deallocation interface.
class SystemMemory {
public:
  virtual void *allocate(size_t size) = 0;
  virtual void *allocate(size_t size, size_t aligned) = 0;
  virtual void deallocate(void *ptr) = 0;
  virtual ~SystemMemory() = default;
};

class USMAllocContext {
public:
  // Keep it public since it needs to be accessed by the lower layer(Buckets)
  class USMAllocImpl;

  USMAllocContext(std::unique_ptr<SystemMemory> memHandle);
  ~USMAllocContext();

  void *allocate(size_t size);
  void *allocate(size_t size, size_t alignment);
  void deallocate(void *ptr);

private:
  std::unique_ptr<USMAllocImpl> pImpl;
};

#endif
