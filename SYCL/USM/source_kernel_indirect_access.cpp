// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %opencl_lib %s -o %t1.out
// RUN: %CPU_RUN_PLACEHOLDER %t1.out
// RUN: %GPU_RUN_PLACEHOLDER %t1.out
// RUN: %ACC_RUN_PLACEHOLDER %t1.out
// REQUIRES: opencl,opencl_icd

#include <CL/cl.h>
#include <sycl/backend/opencl.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;

static const char *Src = R"(
kernel void test(global ulong *PSrc, global ulong *PDst) {
    global int *Src = (global int *) *PSrc;
    global int *Dst = (global int *) *PDst;
    int Old = *Src, New = Old + 1;
    printf("Read %d from %p; write %d to %p\n", Old, Src, New, Dst);
    *Dst = New;
}
)";

int main() {
  queue Q{};

  cl_context Ctx = get_native<backend::opencl>(Q.get_context());
  cl_program Prog = clCreateProgramWithSource(Ctx, 1, &Src, NULL, NULL);
  clBuildProgram(Prog, 0, NULL, NULL, NULL, NULL);

  cl_kernel OclKernel = clCreateKernel(Prog, "test", NULL);

  kernel SyclKernel = make_kernel<backend::opencl>(OclKernel, Q.get_context());

  auto POuter = malloc_shared<int *>(1, Q);
  auto PInner = malloc_shared<int>(1, Q);
  auto QOuter = malloc_shared<int *>(1, Q);
  auto QInner = malloc_shared<int>(1, Q);

  *PInner = 4;
  *POuter = PInner;
  *QInner = 0;
  *QOuter = QInner;

  Q.submit([&](handler &CGH) {
     CGH.set_arg(0, POuter);
     CGH.set_arg(1, QOuter);
     CGH.parallel_for(cl::sycl::range<1>(1), SyclKernel);
   }).wait();

  assert(*PInner == 4 && "Read value is corrupted");
  assert(*QInner == 5 && "Value value is incorrect");

  std::cout << "Increment: " << *PInner << " -> " << *QInner << std::endl;

  clReleaseKernel(OclKernel);
  clReleaseProgram(Prog);
  clReleaseContext(Ctx);
}
