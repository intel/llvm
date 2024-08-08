#include "d.hpp"
#include <iostream>

SYCL_EXTERNAL int levelD(int val) {
#ifndef __SYCL_DEVICE_ONLY__
  std::cerr << "Host symbol used" << std::endl;
  return 0;
#endif
  return val |= (0xD << 12);
}

