// Device pre-linking host-only or kernel-free SYCL inputs must produce an
// object that a host linker can use without linking the SYCL runtime.
// Regression for https://github.com/intel/llvm/issues/23528.
// Use the GNU-style driver options with both clang++ and clang-cl test setups.
// DEFINE: %{cxx} = %clangxx --driver-mode=g++
// Disable automatic SYCL library dependencies in Windows objects as well.
// DEFINE: %{sycl} = %{cxx} -fsycl -nolibsycl
// RUN: %{cxx} -fno-sycl -c %s -o %t.host.o

// No device inputs, with both offload drivers.
// RUN: %{sycl} --no-offload-new-driver -fsycl-link %t.host.o -o %t.old.host.link.o
// RUN: %{cxx} -fno-sycl %t.host.o %t.old.host.link.o -o %t.old.host.out
// RUN: %{run-aux} %t.old.host.out
// RUN: %{sycl} --offload-new-driver -fsycl-link %t.host.o -o %t.new.host.link.o
// RUN: %{cxx} -fno-sycl %t.host.o %t.new.host.link.o -o %t.new.host.out
// RUN: %{run-aux} %t.new.host.out

// Device inputs whose images become empty during post-link processing.
// RUN: %{sycl} --no-offload-new-driver -fgpu-rdc -c %s -o %t.old.sycl.o
// RUN: %{sycl} --no-offload-new-driver -fsycl-link %t.old.sycl.o -o %t.old.sycl.link.o
// RUN: %{cxx} -fno-sycl %t.old.sycl.o %t.old.sycl.link.o -o %t.old.sycl.out
// RUN: %{run-aux} %t.old.sycl.out
// RUN: %{sycl} --offload-new-driver -fgpu-rdc -c %s -o %t.new.sycl.o
// RUN: %{sycl} --offload-new-driver -fsycl-link %t.new.sycl.o -o %t.new.sycl.link.o
// RUN: %{cxx} -fno-sycl %t.new.sycl.o %t.new.sycl.link.o -o %t.new.sycl.out
// RUN: %{run-aux} %t.new.sycl.out

// Without RDC, empty images must still produce the container consumed by host
// compilation, while leaving the resulting object free of runtime references.
// RUN: %{sycl} --no-offload-new-driver -fno-gpu-rdc -c %s -o %t.old.nordc.o
// RUN: %{cxx} -fno-sycl %t.old.nordc.o -o %t.old.nordc.out
// RUN: %{run-aux} %t.old.nordc.out
// RUN: %{sycl} --offload-new-driver -fno-gpu-rdc -c %s -o %t.new.nordc.o
// RUN: %{cxx} -fno-sycl %t.new.nordc.o -o %t.new.nordc.out
// RUN: %{run-aux} %t.new.nordc.out

int main() { return 0; }
