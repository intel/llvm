// Test -fsycl-allow-device-image-dependencies with a single dynamic library on
// Windows and Linux.

// RUN: mkdir -p %t.dir
// RUN: %clangxx --offload-new-driver -fsycl %fPIC %shared_lib -fsycl-allow-device-image-dependencies -I %S/Inputs \
// RUN:    %S/Inputs/a.cpp                                                              \
// RUN:    %S/Inputs/b.cpp                                                              \
// RUN:    %S/Inputs/c.cpp                                                              \
// RUN:    %S/Inputs/d.cpp                                                              \
// RUN:    %S/Inputs/wrapper.cpp                                                        \
// RUN:    -o %if windows %{%t.dir/device_single.dll%} %else %{%t.dir/libdevice_single.so%}

// RUN: %{build} --offload-new-driver -I%S/Inputs -o %t.out           \
// RUN: %if windows                              \
// RUN:   %{%t.dir/device_single.lib%}               \
// RUN: %else                                    \
// RUN:   %{-L%t.dir -ldevice_single -Wl,-rpath=%t.dir%}

// RUN: %{run} %t.out

// XFAIL: target-native_cpu
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/20142

#include "wrapper.hpp"

int main() { return (wrapper()); }
