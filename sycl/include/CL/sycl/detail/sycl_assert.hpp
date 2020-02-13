#include <CL/sycl/detail/defines.hpp>

#include <iostream>
#include <string>

#ifdef __SYCL_ENABLE_ASSERTIONS__
#ifndef __SYCL_DEVICE_ONLY__
#define __SYCL_STR(X) #X
#define __SYCL_XSTR(X) __SYCL_STR(X)
#ifdef WIN32
#define _SEP '\\'
#else
#define _SEP '/'
#endif
#define __SYCL_FILE__                                                          \
  ({                                                                           \
    static const int32_t Idx = sycl::detail::filename(__FILE__);               \
    __FILE__ ":" __SYCL_XSTR(__LINE__) + Idx;                                  \
  })
#define __SYCL_ASSERT(X, ...)                                                  \
  sycl::detail::sycl_assert(__SYCL_FILE__, X, __SYCL_XSTR(X), __VA_ARGS__)

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {
constexpr int32_t filename(const char *const Path, const int32_t Idx = 0,
                           const int32_t SlashIdx = -1) {
  return Path[Idx] ? (Path[Idx] == _SEP ? filename(Path, Idx + 1, Idx)
                                       : filename(Path, Idx + 1, SlashIdx))
                   : SlashIdx + 1;
}
template <typename... Args>
void sycl_assert(const std::string &FileName, bool Cond,
                 const std::string &CondStr, Args &&... Messages) {
  if (!Cond) {
    std::cerr << "Assertion \"" << CondStr << "\" at (" << FileName << "): ";
    using Expander = int[];
    (void)Expander{
        0, (void(std::cerr << std::forward<Args>(Messages) << " "), 0)...};
    exit(-1);
  }
}
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
#undef _SEP
#else
#define __SYCL_ASSERT(X, ...) assert(X)
#endif // #ifdef __SYCL_DEVICE_ONLY__
#else  // #ifdef __SYCL_ENABLE_ASSERTIONS__
#define __SYCL_ASSERT(X, ...)
#endif // #ifdef __SYCL_ENABLE_ASSERTIONS__
