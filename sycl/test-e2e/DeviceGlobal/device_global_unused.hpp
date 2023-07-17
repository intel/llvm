#pragma once

#include "common.hpp"

device_global<int, TestProperties> MemcpyDeviceGlobal;
device_global<int, TestProperties> CopyDeviceGlobal;

int test() {
  queue Q;
  int MemcpyWrite = 42, CopyWrite = 24, MemcpyRead = 1, CopyRead = 2;

  // Copy from device globals before having written anything. This should act as
  // having zero-initialized values.
  Q.memcpy(&MemcpyRead, MemcpyDeviceGlobal);
  Q.copy(CopyDeviceGlobal, &CopyRead);
  Q.wait();
  assert(MemcpyRead == 0);
  assert(CopyRead == 0);

  // Write to device globals and then read their values.
  Q.memcpy(MemcpyDeviceGlobal, &MemcpyWrite);
  Q.copy(&CopyWrite, CopyDeviceGlobal);
  Q.wait();
  Q.memcpy(&MemcpyRead, MemcpyDeviceGlobal);
  Q.copy(CopyDeviceGlobal, &CopyRead);
  Q.wait();
  assert(MemcpyRead == MemcpyWrite);
  assert(CopyRead == CopyWrite);

  return 0;
}
