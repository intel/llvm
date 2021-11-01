// RUN: %clang_cc1 -S -fsycl-is-device -internal-isystem %S/Inputs -triple spir64 -sycl-std=2020 %s

#include "sycl.hpp"

sycl::queue myQueue;

int main() {

  sycl::__mm_host<char> mh;

  myQueue.submit([&](sycl::handler &h) {
    h.single_task<class MmHost>([=] {
      mh.use();
    });
  });

  return 0;
}
