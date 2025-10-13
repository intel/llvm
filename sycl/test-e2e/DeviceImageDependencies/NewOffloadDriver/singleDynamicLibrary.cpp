// Test -fsycl-allow-device-image-dependencies with a single dynamic library on
// Windows and Linux.

// DEFINE: %{tdir} = %t/..
// RUN: mkdir -p %{tdir}
// RUN: %clangxx --offload-new-driver -fsycl %fPIC %shared_lib -fsycl-allow-device-image-dependencies -I %S/Inputs \
// RUN:    %S/Inputs/a.cpp                                                              \
// RUN:    %S/Inputs/b.cpp                                                              \
// RUN:    %S/Inputs/c.cpp                                                              \
// RUN:    %S/Inputs/d.cpp                                                              \
// RUN:    %S/Inputs/wrapper.cpp                                                        \
// RUN:    -o %if windows %{%{tdir}/device_single.dll%} %else %{%{tdir}/libdevice_single.so%}

// RUN: %{build} --offload-new-driver -I%S/Inputs -o %t.out           \
// RUN: %if windows                              \
// RUN:   %{%{tdir}/device_single.lib%}               \
// RUN: %else                                    \
// RUN:   %{-L%{tdir} -ldevice_single -Wl,-rpath=%{tdir}%}

// RUN: %{run} %t.out

// XFAIL: target-native_cpu
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/20142

#include "wrapper.hpp"

int main() { return (wrapper()); }
