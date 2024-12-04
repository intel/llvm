// Test -fsycl-allow-device-image-dependencies with objects.

// UNSUPPORTED: cuda || hip

// RUN: %clangxx -fsycl %S/Inputs/a.cpp -I %S/Inputs -c -o %t_a.o
// RUN: %clangxx -fsycl %S/Inputs/b.cpp -I %S/Inputs -c -o %t_b.o
// RUN: %clangxx -fsycl %S/Inputs/c.cpp -I %S/Inputs -c -o %t_c.o
// RUN: %clangxx -fsycl %S/Inputs/d.cpp -I %S/Inputs -c -o %t_d.o
// RUN: %clangxx -fsycl -fsycl-targets=%{sycl_triple} -fsycl-device-code-split=per_kernel -fsycl-allow-device-image-dependencies %t_a.o %t_b.o %t_c.o %t_d.o %S/Inputs/basic.cpp -o %t.out
// RUN: %{run} %t.out
