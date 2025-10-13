// Test -fsycl-allow-device-image-dependencies with dynamic libraries.

// DEFINE: %{dynamic_lib_options} = -fsycl %fPIC %shared_lib -fsycl-allow-device-image-dependencies -I %S/Inputs %if windows %{-DMAKE_DLL %}
// DEFINE: %{dynamic_lib_suffix} = %if windows %{dll%} %else %{so%}

// DEFINE: %{tdir} = %t/..
// RUN: mkdir -p %{tdir}
// RUN: %clangxx %{dynamic_lib_options} %S/Inputs/d.cpp                                    -o %{tdir}/libdevice_d.%{dynamic_lib_suffix}
// RUN: %clangxx %{dynamic_lib_options} %S/Inputs/c.cpp %if windows %{%{tdir}/libdevice_d.lib%} -o %{tdir}/libdevice_c.%{dynamic_lib_suffix}
// RUN: %clangxx %{dynamic_lib_options} %S/Inputs/b.cpp %if windows %{%{tdir}/libdevice_c.lib%} -o %{tdir}/libdevice_b.%{dynamic_lib_suffix}
// RUN: %clangxx %{dynamic_lib_options} %S/Inputs/a.cpp %if windows %{%{tdir}/libdevice_b.lib%} -o %{tdir}/libdevice_a.%{dynamic_lib_suffix}

// RUN: %clangxx -fsycl %{sycl_target_opts} -fsycl-allow-device-image-dependencies -fsycl-device-code-split=per_kernel %S/Inputs/basic.cpp -o %t.out            \
// RUN: %if windows                                                                       \
// RUN:   %{%{tdir}/libdevice_a.lib%}                                                          \
// RUN: %else                                                                             \
// RUN:   %{-L%{tdir} -ldevice_a -ldevice_b -ldevice_c -ldevice_d -Wl,-rpath=%{tdir}%}

// RUN: %{run} %t.out

// XFAIL: target-native_cpu
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/20142
