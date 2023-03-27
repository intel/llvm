// REQUIRES: opencl, opencl_icd
//
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s %opencl_lib -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <sycl/sycl.hpp>

using namespace sycl;

const char *clSource = R"(
    kernel void test(global int *a, local float *b, int n) {
        if (get_local_id(0) == 0) {
          for (int i = 0; i < n; i++)
              b[i] = i;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        bool ok = true;
        for (int i = 0; i < n; i++)
            ok &= (b[i] == i);

        a[get_global_id(0)] = ok;
    }
)";

kernel getKernel(const queue &Q) {
  cl_int status = CL_SUCCESS;

  auto device_cl = get_native<backend::opencl>(Q.get_device());
  auto context_cl = get_native<backend::opencl>(Q.get_context());
  auto program_cl =
      clCreateProgramWithSource(context_cl, 1, &clSource, nullptr, &status);
  assert(CL_SUCCESS == status);
  status = clBuildProgram(program_cl, 0, nullptr, "", nullptr, nullptr);
  assert(CL_SUCCESS == status);
  auto kernel_cl = clCreateKernel(program_cl, "test", &status);
  assert(CL_SUCCESS == status);

  return make_kernel<backend::opencl>(kernel_cl, Q.get_context());
}

int main() {
  queue Q;
  auto K = getKernel(Q);

  constexpr cl_int N_slm = 256;
  constexpr int N_wg = 32;

  cl_int init[N_wg];
  buffer<cl_int, 1> b(init, N_wg);

  Q.submit([&](handler &cgh) {
    auto acc_global = b.get_access<access::mode::write>(cgh);
    local_accessor<float, 1> acc_local(N_slm, cgh);

    cgh.set_arg(0, acc_global);
    cgh.set_arg(1, acc_local);
    cgh.set_arg(2, N_slm);

    cgh.parallel_for(nd_range<1>(N_wg, 1), K);
  });

  auto acc_global = b.get_access<access::mode::read>();
  for (int i = 0; i < N_wg; i++) {
    if (acc_global[i] != 1) {
      std::cerr << "Error in WG " << i << std::endl;
      exit(1);
    }
  }

  std::cout << "Success" << std::endl;
  return 0;
}
