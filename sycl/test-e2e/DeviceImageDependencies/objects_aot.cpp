// Test -fsycl-allow-device-image-dependencies with objects and AOT.

// REQUIRES: ocloc, gpu

// DEFINE: %{aot_options} = -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen %gpu_aot_target_opts -DUSE_AOT

// RUN: %clangxx %{aot_options} %S/Inputs/a.cpp -I %S/Inputs -c -o %t_a.o
// RUN: %clangxx %{aot_options} %S/Inputs/b.cpp -I %S/Inputs -c -o %t_b.o
// RUN: %clangxx %{aot_options} %S/Inputs/c.cpp -I %S/Inputs -c -o %t_c.o
// RUN: %clangxx %{aot_options} %S/Inputs/d.cpp -I %S/Inputs -c -o %t_d.o
// RUN: %clangxx %{aot_options} -fsycl-device-code-split=per_kernel -fsycl-allow-device-image-dependencies -ftarget-export-symbols %t_a.o %t_b.o %t_c.o %t_d.o %S/Inputs/basic.cpp -o %t.out
// RUN: %{run} %t.out
