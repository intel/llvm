#include <cassert>
#include <iostream>
#include <sycl/sycl.hpp>

using namespace cl::sycl;
static constexpr int MAGIC_NUM = -1;
static constexpr size_t BUFFER_SIZE = 16;

int main(int Argc, const char *Argv[]) {

  sycl::property_list Props{
      sycl::property::queue::in_order{},
      sycl::ext::oneapi::property::queue::discard_events{}};
  sycl::queue Q(Props);

  sycl::range<1> Range(BUFFER_SIZE);
  int *Harray = sycl::malloc_host<int>(BUFFER_SIZE, Q);
  if (Harray == nullptr) {
    return -1;
  }
  for (size_t i = 0; i < BUFFER_SIZE; ++i) {
    Harray[i] = MAGIC_NUM;
  }

  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for<class kernel_using_assert>(
        Range, [=](sycl::item<1> itemID) {
          size_t i = itemID.get_id(0);
          Harray[i] = i + 10;
          assert(Harray[i] == i + 10 && "assert message");
        });
  });
  Q.wait();

  // Checks result
  for (size_t i = 0; i < BUFFER_SIZE; ++i) {
    size_t expected = i + 10;
    if (Harray[i] != expected)
      return -1;
  }
  free(Harray, Q);

  std::cout << "The test passed." << std::endl;
  return 0;
}
