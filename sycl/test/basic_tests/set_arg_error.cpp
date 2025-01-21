// RUN: %clangxx %fsycl-host-only -DUSE_DEPRECATED_LOCAL_ACC -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note,warning %s
// RUN: %clangxx %fsycl-host-only -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note,warning %s

#include <sycl/sycl.hpp>

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
  sycl::buffer<int> buf(size);
  sycl::queue q;
  q.submit([&](sycl::handler &h) {
    auto global_acc = buf.get_access<sycl::access::mode::write>(h);
    sycl::sampler samp(sycl::coordinate_normalization_mode::normalized,
                       sycl::addressing_mode::clamp,
                       sycl::filtering_mode::nearest);
#ifdef USE_DEPRECATED_LOCAL_ACC
    sycl::accessor<int, 1, sycl::access::mode::read_write,
                   sycl::access::target::local>
        local_acc({size}, h);
#else
    sycl::local_accessor<int, 1> local_acc({size}, h);
#endif

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
