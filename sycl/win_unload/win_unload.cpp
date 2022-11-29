#include <iostream>

#include "win_unload.hpp"

// working.
// need namespace, etc.

void init() {
  std::cout << "win_unload init() " << std::endl;
  volatile int x = 31;
}