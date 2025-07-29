// RUN: %clang_cc1 -fsycl-is-device %s -emit-llvm -triple spir64-unknown-unknown -o - | FileCheck %s

struct __attribute__((sycl_special_class))
      [[__sycl_detail__::sycl_type(annotated_arg)]]
    AnnotatedIntPtr {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(
                  "sycl-unaliased", nullptr)]]
              __attribute__((opencl_global)) int* InPtr) {
    Ptr = InPtr;
  }

  int &operator[](unsigned I) const { return Ptr[I]; }

  __attribute__((opencl_global)) int *Ptr;
};

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(const Func &kernelFunc) {
  kernelFunc();
}

int main() {
  {
    int *a;
    int *b;
    int *c;
    kernel<class kernel_nounaliased>([a, b, c]() { c[0] = a[0] + b[0]; });
    // CHECK-DAG: define {{.*}}spir_kernel {{.*}}kernel_nounaliased(ptr addrspace(1) noundef align 4 %{{.*}}, ptr addrspace(1) noundef align 4 %{{.*}}, ptr addrspace(1) noundef align 4 %{{.*}})
  }
  {
    AnnotatedIntPtr a;
    int *b;
    int *c;
    kernel<class kernel_unaliased1>([a, b, c]() { c[0] = a[0] + b[0]; });
    // CHECK-DAG: define {{.*}}spir_kernel {{.*}}kernel_unaliased1(ptr addrspace(1) noalias noundef align 4 "sycl-unaliased" %{{.*}}, ptr addrspace(1) noundef align 4 %{{.*}}, ptr addrspace(1) noundef align 4 %{{.*}})
  }
  {
    int *a;
    AnnotatedIntPtr b;
    int *c;
    kernel<class kernel_unaliased2>([a, b, c]() { c[0] = a[0] + b[0]; });
    // CHECK-DAG: define {{.*}}spir_kernel {{.*}}kernel_unaliased2(ptr addrspace(1) noundef align 4 %{{.*}}, ptr addrspace(1) noalias noundef align 4 "sycl-unaliased" %{{.*}}, ptr addrspace(1) noundef align 4 %{{.*}})
  }
  {
    int *a;
    int *b;
    AnnotatedIntPtr c;
    kernel<class kernel_unaliased3>([a, b, c]() { c[0] = a[0] + b[0]; });
    // CHECK-DAG: define {{.*}}spir_kernel {{.*}}kernel_unaliased3(ptr addrspace(1) noundef align 4 %{{.*}}, ptr addrspace(1) noundef align 4 %{{.*}}, ptr addrspace(1) noalias noundef align 4 "sycl-unaliased" %{{.*}})
  }
  {
    AnnotatedIntPtr a;
    AnnotatedIntPtr b;
    int *c;
    kernel<class kernel_unaliased4>([a, b, c]() { c[0] = a[0] + b[0]; });
    // CHECK-DAG: define {{.*}}spir_kernel {{.*}}kernel_unaliased4(ptr addrspace(1) noalias noundef align 4 "sycl-unaliased" %{{.*}}, ptr addrspace(1) noalias noundef align 4 "sycl-unaliased" %{{.*}}, ptr addrspace(1) noundef align 4 %{{.*}})
  }
  {
    AnnotatedIntPtr a;
    int *b;
    AnnotatedIntPtr c;
    kernel<class kernel_unaliased5>([a, b, c]() { c[0] = a[0] + b[0]; });
    // CHECK-DAG: define {{.*}}spir_kernel {{.*}}kernel_unaliased5(ptr addrspace(1) noalias noundef align 4 "sycl-unaliased" %{{.*}}, ptr addrspace(1) noundef align 4 %{{.*}}, ptr addrspace(1) noalias noundef align 4 "sycl-unaliased" %{{.*}})
  }
  {
    int *a;
    AnnotatedIntPtr b;
    AnnotatedIntPtr c;
    kernel<class kernel_unaliased6>([a, b, c]() { c[0] = a[0] + b[0]; });
    // CHECK-DAG: define {{.*}}spir_kernel {{.*}}kernel_unaliased6(ptr addrspace(1) noundef align 4 %{{.*}}, ptr addrspace(1) noalias noundef align 4 "sycl-unaliased" %{{.*}}, ptr addrspace(1) noalias noundef align 4 "sycl-unaliased" %{{.*}})
  }
  {
    AnnotatedIntPtr a;
    AnnotatedIntPtr b;
    AnnotatedIntPtr c;
    kernel<class kernel_unaliased7>([a, b, c]() { c[0] = a[0] + b[0]; });
    // CHECK-DAG: define {{.*}}spir_kernel {{.*}}kernel_unaliased7(ptr addrspace(1) noalias noundef align 4 "sycl-unaliased" %{{.*}}, ptr addrspace(1) noalias noundef align 4 "sycl-unaliased" %{{.*}}, ptr addrspace(1) noalias noundef align 4 "sycl-unaliased" %{{.*}})
  }
}
