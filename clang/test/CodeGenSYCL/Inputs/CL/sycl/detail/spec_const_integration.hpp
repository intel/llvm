__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

#if __cplusplus >= 201703L
// Translates SYCL 2020 `specialization_id` to a unique symbolic identifier
// which is used internally by the toolchain
template <auto &SpecName> const char *get_spec_constant_symbolic_ID() {
  return get_spec_constant_symbolic_ID_impl<SpecName>();
}
#endif

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
