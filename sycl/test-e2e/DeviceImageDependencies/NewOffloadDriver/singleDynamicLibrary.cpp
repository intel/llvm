// Test -fsycl-allow-device-image-dependencies with a single dynamic library on
// Windows and Linux.

// UNSUPPORTED: cuda || hip
// UNSUPPORTED-INTENDED: Not implemented yet for Nvidia/AMD backends.

// RUN: %clangxx --offload-new-driver -fsycl %fPIC %shared_lib -fsycl-allow-device-image-dependencies -I %S/Inputs \
// RUN:    %S/Inputs/a.cpp                                                              \
// RUN:    %S/Inputs/b.cpp                                                              \
// RUN:    %S/Inputs/c.cpp                                                              \
// RUN:    %S/Inputs/d.cpp                                                              \
// RUN:    %S/Inputs/wrapper.cpp                                                        \
// RUN:    -o %if windows %{%T/device_single.dll%} %else %{%T/libdevice_single.so%}

// RUN: %{build} --offload-new-driver -I%S/Inputs -o %t.out           \
// RUN: %if windows                              \
// RUN:   %{%T/device_single.lib%}               \
// RUN: %else                                    \
// RUN:   %{-L%T -ldevice_single -Wl,-rpath=%T%}

// RUN: %{run} %t.out

#include "wrapper.hpp"

int main() { return (wrapper()); }
