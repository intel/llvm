/*
    // compile to static lib
    clang++ -fsycl -c -fPIC -o simple_lib.o simple_lib.cpp

    // compile to dynamic lib
    clang++ -fsycl  -fPIC -shared -o simple_lib.so simple_lib.cpp

*/

#include <sycl/sycl.hpp>

const size_t BUFF_SIZE = 1;

class Delay {
public:
  std::shared_ptr<sycl::buffer<int, 1>> sharedBuffer;

  void release() {
    std::cout << "Delay.release()" << std::endl;
    sharedBuffer.reset();
  }

  const sycl::buffer<int, 1> &getBuffer() {
    if (!sharedBuffer) {
      sharedBuffer = std::make_shared<sycl::buffer<int, 1>>(BUFF_SIZE);
    }
    return *sharedBuffer;
  }

  Delay() : sharedBuffer(nullptr) {}
  ~Delay() { release(); }
};

Delay *MyDelay = new Delay;

__attribute__((destructor(101))) static void Unload101() {
  std::cout << "lib unload - __attribute__((destructor(101)))" << std::endl;
  delete MyDelay;
}

int add_using_device(int a, int b) {
  sycl::queue q;
  sycl::buffer<int, 1> buf = MyDelay->getBuffer();
  q.submit([&](sycl::handler &cgh) {
     sycl::accessor acc(buf, cgh, sycl::write_only);

     cgh.single_task([=] { acc[0] = a + b; });
   }).wait();

  sycl::host_accessor acc(buf);
  return acc[0];
}
