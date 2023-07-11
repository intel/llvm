// Checks that kernelhandler and subhandler are emitted in the integration
// headers. The sycl-native-cpu helper header is always named
// <sycl-int-header>.hc
// RUN: %clangxx -fsycl-device-only -fsycl-targets=native_cpu -Xclang -fsycl-int-header=%t.h -Xclang -fsycl-int-footer=%t-footer.h -o %t.bc %s
// RUN: FileCheck -input-file=%t-footer.h %s --check-prefix=CHECK-H
// RUN: FileCheck -input-file=%t.h.hc %s --check-prefix=CHECK-HC
// Compiling generated main integration header to check correctness, -fsycl
// option used to find required includes
// RUN: %clangxx -fsycl -D __SYCL_NATIVE_CPU__ -c -x c++ %t.h

#include "sycl.hpp"
class Test1;
int main() {
  sycl::queue deviceQueue;
  sycl::accessor<int, 1, sycl::access::mode::write> acc;
  sycl::range<1> r(1);
  deviceQueue.submit([&](sycl::handler &h) {
    h.parallel_for<Test1>(r, [=](sycl::id<1> id) { acc[id[0]] = 42; });
  });
}

// check that we are including the Native CPU header in the main SYCL ih
// CHECK-H: #include "{{.*}}.h.hc"

// check that we are emitting the subhandler in Native CPU ih
//CHECK-HC: #pragma once
//CHECK-HC-NEXT: #include <sycl/detail/native_cpu.hpp>
//CHECK-HC-NEXT: #include <sycl/detail/pi.h>
//CHECK-HC-NEXT: extern "C" void __sycl_register_lib(pi_device_binaries desc);
//CHECK-HC:extern "C" void _ZTS5Test1_NativeCPUKernelsubhandler(const sycl::detail::NativeCPUArgDesc *MArgs, __nativecpu_state *state);

// check that we are emitting the call to __sycl_register_lib
//CHECK-HC: static _pi_offload_entry_struct _pi_offload_entry_struct_ZTS5Test1_NativeCPUKernel{(void*)&_ZTS5Test1_NativeCPUKernelsubhandler, const_cast<char*>("_ZTS5Test1_NativeCPUKernel"), 1, 0, 0 };
//CHECK-HC-NEXT: static pi_device_binary_struct pi_device_binary_struct_ZTS5Test1_NativeCPUKernel{0, 4, 0, __SYCL_PI_DEVICE_BINARY_TARGET_UNKNOWN, nullptr, nullptr, nullptr, nullptr, (unsigned char*)&_ZTS5Test1_NativeCPUKernelsubhandler, (unsigned char*)&_ZTS5Test1_NativeCPUKernelsubhandler + 1, &_pi_offload_entry_struct_ZTS5Test1_NativeCPUKernel, &_pi_offload_entry_struct_ZTS5Test1_NativeCPUKernel+1, nullptr, nullptr };
//CHECK-HC-NEXT: static pi_device_binaries_struct pi_device_binaries_struct_ZTS5Test1_NativeCPUKernel{0, 1, &pi_device_binary_struct_ZTS5Test1_NativeCPUKernel, nullptr, nullptr };
//CHECK-HC-NEXT: struct init_native_cpu_ZTS5Test1_NativeCPUKernel_t{
//CHECK-HC-NEXT: 	init_native_cpu_ZTS5Test1_NativeCPUKernel_t(){
//CHECK-HC-NEXT: 		__sycl_register_lib(&pi_device_binaries_struct_ZTS5Test1_NativeCPUKernel);
//CHECK-HC-NEXT: 	}
//CHECK-HC-NEXT: };
//CHECK-HC-NEXT: static init_native_cpu_ZTS5Test1_NativeCPUKernel_t init_native_cpu_ZTS5Test1_NativeCPUKernel;
