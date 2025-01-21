// RUN: %clang_cc1 -fsycl-is-device -triple nvptx -internal-isystem %S/Inputs -std=c++2b -verify -Wno-incorrect-sub-group-size %s
// RUN: %clang_cc1 -fsycl-is-device -triple nvptx -internal-isystem %S/Inputs -std=c++2b -verify -Wno-attributes %s
// RUN: %clang_cc1 -fsycl-is-device -triple amdgcn-amd-amdhsa -target-cpu gfx90a -internal-isystem %S/Inputs -std=c++2b -verify -Wno-incorrect-sub-group-size %s
// RUN: %clang_cc1 -fsycl-is-device -triple amdgcn-amd-amdhsa -target-cpu gfx90a -internal-isystem %S/Inputs -std=c++2b -verify -Wno-attributes %s
// RUN: %clang_cc1 -fsycl-is-device -triple amdgcn-amd-amdhsa -target-cpu gfx1010 -internal-isystem %S/Inputs -std=c++2b -verify -Wno-incorrect-sub-group-size %s
// RUN: %clang_cc1 -fsycl-is-device -triple amdgcn-amd-amdhsa -target-cpu gfx1010 -internal-isystem %S/Inputs -std=c++2b -verify -Wno-attributes %s
//
// Sub group size of 8 is incompatible with both CUDA and HIP, expect it to be
// silenced. Check both the dedicated switch '-Wno-incorrect-sub-group-size' and
// the catch all '-Wno-attributes'.
#include "sycl.hpp"


// expected-no-diagnostics
int main() {

  sycl::queue Q;

  Q.submit([&](sycl::handler &h) {
    h.single_task<class invalid_kernel>([=] [[sycl::reqd_sub_group_size(8)]] {});
  });

  return 0;
}
