// REQUIRES: cuda || hip

// RUN: %if cuda %{ %clangxx -fsycl -fsycl-targets=nvptx64-nvidia-cuda -fsycl-device-only -fno-sycl-rdc %s %}
// RUN: %if cuda %{ %clangxx -fsycl -fsycl-targets=nvptx64-nvidia-cuda -fsycl-device-only -fsycl-rdc    %s %}
// RUN: %if hip  %{ %clangxx -fsycl -fsycl-targets=amdgcn-amd-amdhsa   -fsycl-device-only -fno-sycl-rdc %s %}
// RUN: %if hip  %{ %clangxx -fsycl -fsycl-targets=amdgcn-amd-amdhsa   -fsycl-device-only -fsycl-rdc    %s %}

#include <cmath>

int main() {}
