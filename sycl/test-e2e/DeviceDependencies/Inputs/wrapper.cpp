#include <sycl/detail/core.hpp>
#include "a.hpp"
#include <iostream>
#define EXPORT
#include "wrapper.hpp"

using namespace sycl;

class ExeKernel;

int wrapper() {
  int val = 0;
  {
    buffer<int, 1> buf(&val, range<1>(1));
    queue q;
    q.submit([&](handler &cgh) {
      auto acc = buf.get_access(cgh);
      cgh.single_task<ExeKernel>([=]() {acc[0] = levelA(acc[0]);});
    });
  }

  std::cout << "val=" << std::hex << val << "\n";
  if (val!=0xDCBA)
    return (1);  
  return(0);
}
