// RUN: %clangxx %fsycl-host-only -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note,warning %s

#include <CL/sycl.hpp>

struct TriviallyCopyable {
  int a;
  int b;
};

struct NonTriviallyCopyable {
  NonTriviallyCopyable() = default;
  NonTriviallyCopyable(NonTriviallyCopyable const &) {}
  int a;
  int b;
};

struct NonStdLayout {
  int a;

private:
  int b;
};

int main() {
  constexpr size_t size = 1;
  cl::sycl::buffer<int> buf(size);
  cl::sycl::queue q;
  q.submit([&](cl::sycl::handler &h) {
    auto global_acc = buf.get_access<cl::sycl::access::mode::write>(h);
    cl::sycl::sampler samp(cl::sycl::coordinate_normalization_mode::normalized,
                           cl::sycl::addressing_mode::clamp,
                           cl::sycl::filtering_mode::nearest);
    cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                       cl::sycl::access::target::local>
        local_acc({size}, h);
    TriviallyCopyable tc{1, 2};
    NonTriviallyCopyable ntc;
    h.set_arg(0, local_acc);
    h.set_arg(1, global_acc);
    h.set_arg(2, samp);
    h.set_arg(3, tc);
    h.set_arg(4, TriviallyCopyable{});
    h.set_arg( // expected-error {{no matching member function for call to 'set_arg'}}
        5, ntc);
    h.set_arg( // expected-error {{no matching member function for call to 'set_arg'}}
        4, NonTriviallyCopyable{});
#if SYCL_LANGUAGE_VERSION && SYCL_LANGUAGE_VERSION <= 201707
    NonStdLayout nstd;
    h.set_arg( // expected-error {{no matching member function for call to 'set_arg'}}
        6, nstd);
    h.set_arg( // expected-error {{no matching member function for call to 'set_arg'}}
        7, NonStdLayout{});
#endif
  });
}
