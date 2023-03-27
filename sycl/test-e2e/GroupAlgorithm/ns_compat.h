/* https://github.com/intel/llvm/pull/2231: namespace compatibility mode*/
__SYCL_INLINE_NAMESPACE(cl){namespace sycl{namespace intel{};
namespace ext::oneapi {
using namespace sycl::intel;
}
}
}
;
