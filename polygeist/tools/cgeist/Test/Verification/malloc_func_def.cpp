// RUN: not cgeist %s -O0 --function=* -S -emit-llvm 2>&1 | FileCheck %s

// CHECK: error: 'malloc' with a definition might not be compatible with stdlib function

#include <cstdint>

void *f0(std::size_t size) { return new uint8_t(size); }

extern "C" void *malloc(std::size_t size) {
  return new uint8_t(size);
}

void *f1(std::size_t size) { return malloc(size); }
