// for CUDA and HIP the failure happens at compile time, not during runtime
// UNSUPPORTED: cuda || hip || ze_debug

// RUN: %{build} -DSYCL_DISABLE_FALLBACK_ASSERT=1 -o %t.out
// RUN: %{build} -DSYCL_DISABLE_FALLBACK_ASSERT=1 -DGPU -o %t_gpu.out
// RUN: env SYCL_CACHE_PERSISTENT=1 %{run} %if gpu %{ %t_gpu.out %} %else %{ %t.out %}

#include <sycl/detail/core.hpp>

SYCL_EXTERNAL
void undefined();

void test() {
  sycl::queue Queue;

  auto Kernel = []() {
#ifdef __SYCL_DEVICE_ONLY__
#ifdef GPU
    asm volatile("undefined\n");
#else  // GPU
    undefined();
#endif // GPU
#endif // __SYCL_DEVICE_ONLY__
  };

  std::string Msg;
  int Result;

  for (int Idx = 0; Idx < 2; ++Idx) {
    try {
      Queue.submit([&](sycl::handler &CGH) {
        CGH.single_task<class SingleTask>(Kernel);
      });
      assert(false && "There must be compilation error");
    } catch (const sycl::compile_program_error &e) {
      fprintf(stderr, "Exception: %s, %d\n", e.what(), e.get_cl_code());
      if (Idx == 0) {
        Msg = e.what();
        Result = e.get_cl_code();
      } else {
        // Exception constantly adds info on its error code in the message
        assert(Msg.find_first_of(e.what()) == 0 &&
               "PI_ERROR_BUILD_PROGRAM_FAILURE");
        assert(Result == e.get_cl_code() && "Exception code differs");
      }
    } catch (...) {
      assert(false && "There must be sycl::compile_program_error");
    }
  }
}

int main() {
  test();

  return 0;
}
