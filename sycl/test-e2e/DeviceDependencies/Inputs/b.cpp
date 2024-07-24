#include "b.hpp"
#include "c.hpp"

SYCL_EXTERNAL int levelB(int val) {
  val=levelC(val);  
  return val|=(0xB<<4);
}

