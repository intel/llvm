#include <iostream>
#include "a.hpp"
#include "b.hpp"

SYCL_EXTERNAL int levelA(int val) {
#ifndef __SYCL_DEVICE_ONLY__
  std::cerr << "Host symbol used" << std::endl;
  return 0;
#endif
  val=levelB(val);
  return val|=(0xA<<0);
}

