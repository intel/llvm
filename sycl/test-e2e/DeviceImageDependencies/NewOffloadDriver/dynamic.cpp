// Test -fsycl-allow-device-image-dependencies with dynamic libraries.

// DEFINE: %{dynamic_lib_options} = -fsycl %fPIC %shared_lib -fsycl-allow-device-image-dependencies -I %S/Inputs %if windows %{-DMAKE_DLL %}
// DEFINE: %{dynamic_lib_suffix} = %if windows %{dll%} %else %{so%}

// DEFINE: %{tdir} = %t/..
// RUN: rm -rf %{tdir}; mkdir -p %{tdir}
// RUN: %clangxx --offload-new-driver %{dynamic_lib_options} %S/Inputs/d.cpp                                    -o %{tdir}/libdevice_d.%{dynamic_lib_suffix}
// RUN: %clangxx --offload-new-driver %{dynamic_lib_options} %S/Inputs/c.cpp %if windows %{%{tdir}/libdevice_d.lib%} -o %{tdir}/libdevice_c.%{dynamic_lib_suffix}
// RUN: %clangxx --offload-new-driver %{dynamic_lib_options} %S/Inputs/b.cpp %if windows %{%{tdir}/libdevice_c.lib%} -o %{tdir}/libdevice_b.%{dynamic_lib_suffix}
// RUN: %clangxx --offload-new-driver %{dynamic_lib_options} %S/Inputs/a.cpp %if windows %{%{tdir}/libdevice_b.lib%} -o %{tdir}/libdevice_a.%{dynamic_lib_suffix}

// RUN: %{build} --offload-new-driver -fsycl-allow-device-image-dependencies -I %S/Inputs -o %t.out            \
// RUN: %if windows                                                                       \
// RUN:   %{%{tdir}/libdevice_a.lib%}                                                          \
// RUN: %else                                                                             \
// RUN:   %{-L%{tdir} -ldevice_a -ldevice_b -ldevice_c -ldevice_d -Wl,-rpath=%{tdir}%}

// RUN: %{run} %t.out

// XFAIL: target-native_cpu
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/20142

#include "a.hpp"
#include <iostream>
#include <sycl/detail/core.hpp>

using namespace sycl;

class ExeKernel;

int main() {
  int val = 0;
  {
    buffer<int, 1> buf(&val, range<1>(1));
    queue q;
    q.submit([&](handler &cgh) {
      auto acc = buf.get_access(cgh);
      cgh.single_task<ExeKernel>([=]() { acc[0] = levelA(acc[0]); });
    });
  }

  std::cout << "val=" << std::hex << val << "\n";
  if (val != 0xDCBA)
    return (1);
  return (0);
}
