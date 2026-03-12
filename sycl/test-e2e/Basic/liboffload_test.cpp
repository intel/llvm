  #include <iostream>
  #include <sycl/sycl.hpp>
  using namespace sycl;  // (optional) avoids need for "sycl::" before SYCL names



  int main() {
   //  Create a default queue to enqueue work to the default device
   queue myQueue;



   // Allocate shared memory bound to the device and context associated to the
   // queue Replacing malloc_shared with malloc_host would yield a correct
   // program that allocated device-visible memory on the host.
   int* data = sycl::malloc_shared<int>(1024, myQueue);



   myQueue.parallel_for(1024, [=](id<1> idx) {
     // Initialize each buffer element with its own rank number starting at 0
     data[idx] = idx;
   });  // End of the kernel function



   // Explicitly wait for kernel execution since there is no accessor involved
   myQueue.wait();



   // Print result
   for (int i = 0; i < 1024; i++)
     std::cout << "data[" << i << "] = " << data[i] << std::endl;



   sycl::free(data, myQueue);



   return 0;
  }