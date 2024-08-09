// Test -fsycl-allow-device-dependencies with dynamic libraries.

// UNSUPPORTED: cuda || hip

// DEFINE: %{dynamic_lib_options} = -fsycl %fPIC %shared_lib -fsycl-allow-device-dependencies -I %S/Inputs %if windows %{-DMAKE_DLL %}
// DEFINE: %{dynamic_lib_suffix} = %if windows %{dll%} %else %{so%}

// RUN: %clangxx %{dynamic_lib_options} %S/Inputs/d.cpp                                    -o %T/libdevice_d.%{dynamic_lib_suffix}
// RUN: %clangxx %{dynamic_lib_options} %S/Inputs/c.cpp %if windows %{%T/libdevice_d.lib%} -o %T/libdevice_c.%{dynamic_lib_suffix}
// RUN: %clangxx %{dynamic_lib_options} %S/Inputs/b.cpp %if windows %{%T/libdevice_c.lib%} -o %T/libdevice_b.%{dynamic_lib_suffix}
// RUN: %clangxx %{dynamic_lib_options} %S/Inputs/a.cpp %if windows %{%T/libdevice_b.lib%} -o %T/libdevice_a.%{dynamic_lib_suffix}

// RUN: %{build} -fsycl-allow-device-dependencies -I %S/Inputs -o %t.out                  \
// RUN: %if windows                                                                       \
// RUN:   %{%T/libdevice_a.lib%}                                                          \
// RUN: %else                                                                             \
// RUN:   %{-L%T -ldevice_a -ldevice_b -ldevice_c -ldevice_d -Wl,-rpath=%T%}

// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
#include "a.hpp"
#include <iostream>

using namespace sycl;

class ExeKernel;

int main() {
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
