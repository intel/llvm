// Test device image linking when using dynamic libraries when one of
// the device image is compressed and the other is not.

// REQUIRES: zstd

// DEFINE: %{dynamic_lib_options} = -fsycl %fPIC %shared_lib -fsycl-allow-device-image-dependencies -I %S/Inputs %if windows %{-DMAKE_DLL %}
// DEFINE: %{dynamic_lib_suffix} = %if windows %{dll%} %else %{so%}

// RUN: %clangxx %{dynamic_lib_options} %S/Inputs/d.cpp \
// RUN:   -o %T/libdevicecompress_d.%{dynamic_lib_suffix}

// RUN: %clangxx %{dynamic_lib_options} %S/Inputs/c.cpp \
// RUN:   %if windows %{%T/libdevicecompress_d.lib%} \
// RUN:   -o %T/libdevicecompress_c.%{dynamic_lib_suffix}

// RUN: %clangxx %{dynamic_lib_options} %S/Inputs/b.cpp \
// RUN:   %if windows %{%T/libdevicecompress_c.lib%} \
// RUN:   -o %T/libdevicecompress_b.%{dynamic_lib_suffix}

// RUN: %clangxx %{dynamic_lib_options} %S/Inputs/a.cpp \
// RUN:   %if windows %{%T/libdevicecompress_b.lib%} \
// RUN:   -o %T/libdevicecompress_a.%{dynamic_lib_suffix}

// Compressed main executable, while dependencies are not compressed.

// RUN: %clangxx -fsycl --offload-compress %{sycl_target_opts} \
// RUN:   -fsycl-allow-device-image-dependencies -fsycl-device-code-split=per_kernel      \
// RUN:   %S/Inputs/basic.cpp -o %t.out                                                   \
// RUN:   %if windows                                                                     \
// RUN:     %{%T/libdevicecompress_a.lib%}                                                \
// RUN:   %else                                                                           \
// RUN:     %{-L%T -ldevicecompress_a -ldevicecompress_b -ldevicecompress_c -ldevicecompress_d -Wl,-rpath=%T%}

// RUN: %{run} %t.out
