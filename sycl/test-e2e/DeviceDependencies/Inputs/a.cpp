#include "a.hpp"
#include "b.hpp"

SYCL_EXTERNAL int levelA(int val) {
  val=levelB(val);
  return val|=(0xA<<0);
}

