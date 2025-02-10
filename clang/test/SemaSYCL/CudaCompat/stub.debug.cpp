// RUN: %clang_cc1 %s -fsycl-is-host -aux-triple nvptx64-nvidia-cuda -debug-info-kind=constructor -dwarf-version=5 -fsycl-cuda-compatibility -emit-llvm -O3 -o %t

__attribute__((device)) void callee() {}

void caller() {
  callee();
}