#define C_EXPORT
#include "c.hpp"
#include "d.hpp"
#include <iostream>

C_DECLSPEC SYCL_EXTERNAL int levelC(int val) {
#ifndef __SYCL_DEVICE_ONLY__
  std::cerr << "Host symbol used" << std::endl;
  val ^= 0x3456;
#endif
  val=levelD(val);  
  return val|=(0xC<<8);
}
