#include "c.hpp"
#include "d.hpp"

SYCL_EXTERNAL int levelC(int val) {
  val=levelD(val);  
  return val|=(0xC<<8);
}

