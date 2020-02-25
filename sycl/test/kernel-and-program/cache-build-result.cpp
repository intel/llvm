// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out

#include <CL/sycl.hpp>

SYCL_EXTERNAL
void undefined();

void test() {
  cl::sycl::queue Queue;

  auto Kernel = []() {
#ifdef __SYCL_DEVICE_ONLY__
    undefined();
#endif
  };

  std::string Msg;
  int Result;

  for (int Idx = 0; Idx < 2; ++Idx) {
    try {
      Queue.submit([&](cl::sycl::handler &CGH) {
        CGH.single_task<class SingleTask>(Kernel);
      });
      assert(false && "There must be compilation error");
    } catch (const cl::sycl::compile_program_error &e) {
      fprintf(stderr, "Exception: %s, %d\n", e.what(), e.get_cl_code());
      if (Idx == 0) {
        Msg = e.what();
        Result = e.get_cl_code();
      } else {
        // Exception constantly adds info on its error code in the message
        assert(Msg.find_first_of(e.what()) == 0 && "Exception text differs");
        assert(Result == e.get_cl_code() && "Exception code differs");
      }
    } catch (...) {
      assert(false && "There must be cl::sycl::compile_program_error");
    }
  }
}

int main() {
  test();

  return 0;
}
