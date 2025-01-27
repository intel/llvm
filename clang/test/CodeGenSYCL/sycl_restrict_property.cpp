// RUN: %clang_cc1 -fsycl-is-device %s -emit-llvm -triple spir64-unknown-unknown -o - | FileCheck %s

struct __attribute__((sycl_special_class))
      [[__sycl_detail__::sycl_type(annotated_arg)]]
    AnnotatedIntPtr {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(
                  "sycl-restrict", nullptr)]]
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
    kernel<class kernel_norestrict>([a, b, c]() { c[0] = a[0] + b[0]; });
    // CHECK-DAG: define {{.*}}spir_kernel {{.*}}kernel_norestrict(ptr addrspace(1) noundef align 4 %{{.*}}, ptr addrspace(1) noundef align 4 %{{.*}}, ptr addrspace(1) noundef align 4 %{{.*}})
  }
  {
    AnnotatedIntPtr a;
    int *b;
    int *c;
    kernel<class kernel_restrict1>([a, b, c]() { c[0] = a[0] + b[0]; });
    // CHECK-DAG: define {{.*}}spir_kernel {{.*}}kernel_restrict1(ptr addrspace(1) noalias noundef align 4 "sycl-restrict" %{{.*}}, ptr addrspace(1) noundef align 4 %{{.*}}, ptr addrspace(1) noundef align 4 %{{.*}})
  }
  {
    int *a;
    AnnotatedIntPtr b;
    int *c;
    kernel<class kernel_restrict2>([a, b, c]() { c[0] = a[0] + b[0]; });
    // CHECK-DAG: define {{.*}}spir_kernel {{.*}}kernel_restrict2(ptr addrspace(1) noundef align 4 %{{.*}}, ptr addrspace(1) noalias noundef align 4 "sycl-restrict" %{{.*}}, ptr addrspace(1) noundef align 4 %{{.*}})
  }
  {
    int *a;
    int *b;
    AnnotatedIntPtr c;
    kernel<class kernel_restrict3>([a, b, c]() { c[0] = a[0] + b[0]; });
    // CHECK-DAG: define {{.*}}spir_kernel {{.*}}kernel_restrict3(ptr addrspace(1) noundef align 4 %{{.*}}, ptr addrspace(1) noundef align 4 %{{.*}}, ptr addrspace(1) noalias noundef align 4 "sycl-restrict" %{{.*}})
  }
  {
    AnnotatedIntPtr a;
    AnnotatedIntPtr b;
    int *c;
    kernel<class kernel_restrict4>([a, b, c]() { c[0] = a[0] + b[0]; });
    // CHECK-DAG: define {{.*}}spir_kernel {{.*}}kernel_restrict4(ptr addrspace(1) noalias noundef align 4 "sycl-restrict" %{{.*}}, ptr addrspace(1) noalias noundef align 4 "sycl-restrict" %{{.*}}, ptr addrspace(1) noundef align 4 %{{.*}})
  }
  {
    AnnotatedIntPtr a;
    int *b;
    AnnotatedIntPtr c;
    kernel<class kernel_restrict5>([a, b, c]() { c[0] = a[0] + b[0]; });
    // CHECK-DAG: define {{.*}}spir_kernel {{.*}}kernel_restrict5(ptr addrspace(1) noalias noundef align 4 "sycl-restrict" %{{.*}}, ptr addrspace(1) noundef align 4 %{{.*}}, ptr addrspace(1) noalias noundef align 4 "sycl-restrict" %{{.*}})
  }
  {
    int *a;
    AnnotatedIntPtr b;
    AnnotatedIntPtr c;
    kernel<class kernel_restrict6>([a, b, c]() { c[0] = a[0] + b[0]; });
    // CHECK-DAG: define {{.*}}spir_kernel {{.*}}kernel_restrict6(ptr addrspace(1) noundef align 4 %{{.*}}, ptr addrspace(1) noalias noundef align 4 "sycl-restrict" %{{.*}}, ptr addrspace(1) noalias noundef align 4 "sycl-restrict" %{{.*}})
  }
  {
    AnnotatedIntPtr a;
    AnnotatedIntPtr b;
    AnnotatedIntPtr c;
    kernel<class kernel_restrict7>([a, b, c]() { c[0] = a[0] + b[0]; });
    // CHECK-DAG: define {{.*}}spir_kernel {{.*}}kernel_restrict7(ptr addrspace(1) noalias noundef align 4 "sycl-restrict" %{{.*}}, ptr addrspace(1) noalias noundef align 4 "sycl-restrict" %{{.*}}, ptr addrspace(1) noalias noundef align 4 "sycl-restrict" %{{.*}})
  }
}
