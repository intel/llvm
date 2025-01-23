#include "mypipe.hpp"
#include <cstdint>

void KernelFunctor::operator()() const {
  uint32_t data = 2;
  my_pipe::write(data);
}
