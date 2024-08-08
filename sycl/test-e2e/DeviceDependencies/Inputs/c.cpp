#include "c.hpp"
#include "d.hpp"
#include <iostream>

SYCL_EXTERNAL int levelC(int val) {
#ifndef __SYCL_DEVICE_ONLY__
  std::cerr << "Host symbol used" << std::endl;
  return 0;
#endif
  val = levelD(val);
  return val |= (0xC << 8);
}

