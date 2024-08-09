#define D_EXPORT
#include "d.hpp"
#include <iostream>

D_DECLSPEC SYCL_EXTERNAL int levelD(int val) {
#ifndef __SYCL_DEVICE_ONLY__
  std::cerr << "Host symbol used" << std::endl;
  val ^= 0x4567;
#endif
  return val|=(0xD<<12);
}
