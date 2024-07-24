#include "d.hpp"

SYCL_EXTERNAL int levelD(int val) {
  return val|=(0xD<<12);
}

