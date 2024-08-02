#include <iostream>
#include "b.hpp"
#include "c.hpp"

SYCL_EXTERNAL int levelB(int val) {
#ifndef __SYCL_DEVICE_ONLY__
  std::cerr << "Host symbol used" << std::endl;
  return 0;
#endif
  val=levelC(val);  
  return val|=(0xB<<4);
}

