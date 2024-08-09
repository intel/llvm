#define B_EXPORT
#include "b.hpp"
#include "c.hpp"
#include <iostream>

B_DECLSPEC SYCL_EXTERNAL int levelB(int val) {
#ifndef __SYCL_DEVICE_ONLY__
  std::cerr << "Host symbol used" << std::endl;
  val ^= 0x2345;
#endif
  val=levelC(val);  
  return val|=(0xB<<4);
}
