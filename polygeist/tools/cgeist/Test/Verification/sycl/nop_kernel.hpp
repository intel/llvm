// This is an auxiliary file to be used when testing raising.
// It provides a NOP kernel launch in host code that will trigger raising.

void do_nothing(sycl::queue q) {
  q.submit([&](sycl::handler &cgh) {
    cgh.single_task<class K>([=]() {});
  });
}
