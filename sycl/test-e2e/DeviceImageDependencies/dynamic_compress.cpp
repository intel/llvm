// Test device image linking when using dynamic libraries when one of
// the device image is compressed and the other is not.

// REQUIRES: zstd

// XFAIL: target-native_cpu
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/20397

// DEFINE: %{dynamic_lib_options} = -fsycl %fPIC %shared_lib -fsycl-allow-device-image-dependencies -I %S/Inputs %if windows %{-DMAKE_DLL %}
// DEFINE: %{dynamic_lib_suffix} = %if windows %{dll%} %else %{so%}

// RUN: rm -rf %t.dir; mkdir -p %t.dir
// RUN: %clangxx %{dynamic_lib_options} %S/Inputs/d.cpp \
// RUN:   -o %t.dir/libdevicecompress_d.%{dynamic_lib_suffix}

// RUN: %clangxx %{dynamic_lib_options} %S/Inputs/c.cpp \
// RUN:   %if windows %{%t.dir/libdevicecompress_d.lib%} \
// RUN:   -o %t.dir/libdevicecompress_c.%{dynamic_lib_suffix}

// RUN: %clangxx %{dynamic_lib_options} %S/Inputs/b.cpp \
// RUN:   %if windows %{%t.dir/libdevicecompress_c.lib%} \
// RUN:   -o %t.dir/libdevicecompress_b.%{dynamic_lib_suffix}

// RUN: %clangxx %{dynamic_lib_options} %S/Inputs/a.cpp \
// RUN:   %if windows %{%t.dir/libdevicecompress_b.lib%} \
// RUN:   -o %t.dir/libdevicecompress_a.%{dynamic_lib_suffix}

// Compressed main executable, while dependencies are not compressed.

// RUN: %clangxx -fsycl --offload-compress %{sycl_target_opts} \
// RUN:   -fsycl-allow-device-image-dependencies -fsycl-device-code-split=per_kernel      \
// RUN:   %S/Inputs/basic.cpp -o %t.out                                                   \
// RUN:   %if windows                                                                     \
// RUN:     %{%t.dir/libdevicecompress_a.lib %t.dir/libdevicecompress_b.lib %t.dir/libdevicecompress_c.lib %t.dir/libdevicecompress_d.lib%} \
// RUN:   %else                                                                           \
// RUN:     %{-L%t.dir -ldevicecompress_a -ldevicecompress_b -ldevicecompress_c -ldevicecompress_d -Wl,-rpath=%t.dir%}

// RUN: %if windows %{ cmd /c "set PATH=%t.dir;%PATH% && %{run} %t.out" %} %else %{ %{run} %t.out %}
