// REQUIRES: opencl-aot, opencl-cpu-rt

// RUN: %{run-aux} %clangxx -fsycl -fsycl-targets=spir64_x86_64 %S/Inputs/is_compatible_with_env.cpp -o %t.out

// RUN: %if !cpu %{ not %} %{run} %t.out
