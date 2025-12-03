// Test -fsycl-allow-device-image-dependencies with dynamic libraries.

// DEFINE: %{dynamic_lib_options} = -fsycl %fPIC %shared_lib -fsycl-allow-device-image-dependencies -I %S/Inputs %if windows %{-DMAKE_DLL %}
// DEFINE: %{dynamic_lib_suffix} = %if windows %{dll%} %else %{so%}

// RUN: rm -rf %t.dir; mkdir -p %t.dir
// RUN: %clangxx --offload-new-driver %{dynamic_lib_options} %S/Inputs/d.cpp                                    -o %t.dir/libdevice_d.%{dynamic_lib_suffix}
// RUN: %clangxx --offload-new-driver %{dynamic_lib_options} %S/Inputs/c.cpp %if windows %{%t.dir/libdevice_d.lib%} -o %t.dir/libdevice_c.%{dynamic_lib_suffix}
// RUN: %clangxx --offload-new-driver %{dynamic_lib_options} %S/Inputs/b.cpp %if windows %{%t.dir/libdevice_c.lib%} -o %t.dir/libdevice_b.%{dynamic_lib_suffix}
// RUN: %clangxx --offload-new-driver %{dynamic_lib_options} %S/Inputs/a.cpp %if windows %{%t.dir/libdevice_b.lib%} -o %t.dir/libdevice_a.%{dynamic_lib_suffix}

// RUN: %{build} --offload-new-driver -fsycl-allow-device-image-dependencies -I %S/Inputs -o %t.dir/%{t:stem}.out            \
// RUN: %if windows                                                                       \
// RUN:   %{%t.dir/libdevice_a.lib%}                                                          \
// RUN: %else                                                                             \
// RUN:   %{-L%t.dir -ldevice_a -ldevice_b -ldevice_c -ldevice_d -Wl,-rpath=%t.dir%}

// RUN: %{run} %t.dir/%{t:stem}.out

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
