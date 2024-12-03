#include "declarations.hpp"

void construct(sycl::queue Q, storage_t *DeviceStorage, unsigned TestCase) {
  Q.submit([&](sycl::handler &CGH) {
     CGH.single_task([=]() {
       DeviceStorage->construct</* ret type = */ BaseIncrement>(TestCase, 19,
                                                                23);
     });
   }).wait_and_throw();
}
