// Test -fsycl-allow-device-image-dependencies with dynamic libraries.

// UNSUPPORTED: cuda || hip

// DEFINE: %{dynamic_lib_options} = -fsycl %fPIC %shared_lib -fsycl-allow-device-image-dependencies -I %S/Inputs %if windows %{-DMAKE_DLL %}
// DEFINE: %{dynamic_lib_suffix} = %if windows %{dll%} %else %{so%}

// RUN: %clangxx %{dynamic_lib_options} %S/Inputs/d.cpp                                    -o %T/libdevice_d.%{dynamic_lib_suffix}
// RUN: %clangxx %{dynamic_lib_options} %S/Inputs/c.cpp %if windows %{%T/libdevice_d.lib%} -o %T/libdevice_c.%{dynamic_lib_suffix}
// RUN: %clangxx %{dynamic_lib_options} %S/Inputs/b.cpp %if windows %{%T/libdevice_c.lib%} -o %T/libdevice_b.%{dynamic_lib_suffix}
// RUN: %clangxx %{dynamic_lib_options} %S/Inputs/a.cpp %if windows %{%T/libdevice_b.lib%} -o %T/libdevice_a.%{dynamic_lib_suffix}

// RUN: %clangxx -fsycl -fsycl-targets=%{sycl_triple} -fsycl-allow-device-image-dependencies -fsycl-device-code-split=per_kernel %S/Inputs/basic.cpp -o %t.out            \
// RUN: %if windows                                                                       \
// RUN:   %{%T/libdevice_a.lib%}                                                          \
// RUN: %else                                                                             \
// RUN:   %{-L%T -ldevice_a -ldevice_b -ldevice_c -ldevice_d -Wl,-rpath=%T%}

// RUN: %{run} %t.out
