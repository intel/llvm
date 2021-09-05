// RUN: %clangxx %fsycl-host-only -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s -o %t.out

#include <CL/sycl.hpp>
#include <CL/sycl/backend/opencl.hpp>

using namespace sycl;

int main() {
  // expected-warning@+1 {{kernel_work_group enumeration is deprecated, use SYCL 2020 requests instead}}
  info::kernel_work_group X;

  context C;
  cl_kernel K1;
  device D;
  kernel K = make_kernel<backend::opencl>(K1, C);
  // once for function, the other one is for function specialization
  // expected-warning@+5 {{get_work_group_info() is deprecated, use SYCL 2020 kernel_device_specific queries instead}}
  // expected-warning@+4 {{get_work_group_info() is deprecated, use SYCL 2020 kernel_device_specific queries instead}}
  // once for enum, the other one is for enum element
  // expected-warning@+2 {{kernel_work_group enumeration is deprecated, use SYCL 2020 requests instead}}
   // expected-warning@+1 {{kernel_work_group enumeration is deprecated, use SYCL 2020 requests instead}}
  (void)K.get_work_group_info<info::kernel_work_group::global_work_size>(D);

  return 0;
}
