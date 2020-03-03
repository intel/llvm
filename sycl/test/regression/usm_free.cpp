// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out

//==-------------- usm_free.cpp - SYCL USM free malloc_shared and free test -------------==//
//
// This test checks if users will successfully allocate 160, 0, and -16 bytes of shared
// memory, and also test user can call free() without worrying about nullptr or invalid
// memory descriptor returned from malloc.
//==-------------------------------------------------------------------------------------==//

#include <iostream>
#include <CL/sycl.hpp>
#include <stdlib.h>
using namespace cl::sycl;

int main(int argc, char * argv[]) {


  int rows = 5, rank = 2;

  auto exception_handler = [](cl::sycl::exception_list exceptions) {
        for (std::exception_ptr const &e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (cl::sycl::exception const &e) {
                std::cout << "Caught asynchronous SYCL "
                             "exception during sparse::trsv:\n"
                          << e.what() << std::endl;
            }
        }
    };

  cl::sycl::device dev = cl::sycl::device(cl::sycl::gpu_selector());
  cl::sycl::queue main_queue(dev, exception_handler);

  double *ia = (double *)malloc_shared(160, main_queue.get_device(), main_queue.get_context());
  double *ja = (double *)malloc_shared(0, main_queue.get_device(), main_queue.get_context());
  double *result = (double *)malloc_shared(-16, main_queue.get_device(), main_queue.get_context());

  std::cout<<"ia : "<<ia<<" ja: "<<ja<<" result : "<<result<<std::endl;

  //throws CL_INVALID_VALUE
  free(ia, main_queue.get_context());
  free(nullptr);
  free(ja, main_queue.get_context());
  free(result, main_queue.get_context());

  return 0;
}
