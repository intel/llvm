#define A_EXPORT
#include "a.hpp"
#include "b.hpp"
#include <iostream>

A_DECLSPEC SYCL_EXTERNAL int levelA(int val) {
#ifndef __SYCL_DEVICE_ONLY__
  std::cerr << "Host symbol used" << std::endl;
  return 0;
#endif
  val = levelB(val);
  return val |= (0xA << 0);
}

