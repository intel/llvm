// RUN: %{build} %s
#include <sycl/ext/codeplay/usm_props.hpp>
#include <sycl/sycl.hpp>
#undef NDEBUG

#include <cassert>

#ifndef SYCL_EXT_CODEPLAY_USM_PROPS
#error SYCL_EXT_CODEPLAY_USM_PROPS is not defined
#endif
#if SYCL_EXT_CODEPLAY_USM_PROPS != 1
#error SYCL_EXT_CODEPLAY_USM_PROPS has unexpected value
#endif

#define N (1024)

// check property requirements from
// https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#table.members.propertyinterface
// Avoid wasting space by just using the `_v` suffixed variant which
// depends on the un-suffixed version

static_assert(sycl::is_property_v<sycl::ext::codeplay::usm_props::host_hot>);
static_assert(
    sycl::is_property_of_v<sycl::ext::codeplay::usm_props::host_hot,
                           sycl::usm_allocator<int, sycl::usm::alloc::host>>);

static_assert(sycl::is_property_v<sycl::ext::codeplay::usm_props::device_hot>);
static_assert(
    sycl::is_property_of_v<sycl::ext::codeplay::usm_props::device_hot,
                           sycl::usm_allocator<int, sycl::usm::alloc::host>>);

static_assert(sycl::is_property_v<sycl::ext::codeplay::usm_props::host_cold>);
static_assert(
    sycl::is_property_of_v<sycl::ext::codeplay::usm_props::host_cold,
                           sycl::usm_allocator<int, sycl::usm::alloc::host>>);

static_assert(sycl::is_property_v<sycl::ext::codeplay::usm_props::device_cold>);
static_assert(
    sycl::is_property_of_v<sycl::ext::codeplay::usm_props::device_cold,
                           sycl::usm_allocator<int, sycl::usm::alloc::host>>);

static_assert(sycl::is_property_v<
              sycl::ext::codeplay::usm_props::host_cache_non_coherent>);
static_assert(sycl::is_property_of_v<
              sycl::ext::codeplay::usm_props::host_cache_non_coherent,
              sycl::usm_allocator<int, sycl::usm::alloc::host>>);

static_assert(sycl::is_property_v<
              sycl::ext::codeplay::usm_props::device_cache_non_coherent>);
static_assert(sycl::is_property_of_v<
              sycl::ext::codeplay::usm_props::device_cache_non_coherent,
              sycl::usm_allocator<int, sycl::usm::alloc::host>>);

static_assert(sycl::is_property_v<
              sycl::ext::codeplay::usm_props::host_cache_write_combine>);
static_assert(sycl::is_property_of_v<
              sycl::ext::codeplay::usm_props::host_cache_write_combine,
              sycl::usm_allocator<int, sycl::usm::alloc::host>>);

static_assert(sycl::is_property_v<
              sycl::ext::codeplay::usm_props::device_cache_write_combine>);
static_assert(sycl::is_property_of_v<
              sycl::ext::codeplay::usm_props::device_cache_write_combine,
              sycl::usm_allocator<int, sycl::usm::alloc::host>>);

static_assert(sycl::is_property_v<
              sycl::ext::codeplay::usm_props::host_access_sequential>);
static_assert(sycl::is_property_of_v<
              sycl::ext::codeplay::usm_props::host_access_sequential,
              sycl::usm_allocator<int, sycl::usm::alloc::host>>);

static_assert(sycl::is_property_v<
              sycl::ext::codeplay::usm_props::device_access_sequential>);
static_assert(sycl::is_property_of_v<
              sycl::ext::codeplay::usm_props::device_access_sequential,
              sycl::usm_allocator<int, sycl::usm::alloc::host>>);

static_assert(
    sycl::is_property_v<sycl::ext::codeplay::usm_props::host_access_random>);
static_assert(
    sycl::is_property_of_v<sycl::ext::codeplay::usm_props::host_access_random,
                           sycl::usm_allocator<int, sycl::usm::alloc::host>>);

static_assert(
    sycl::is_property_v<sycl::ext::codeplay::usm_props::device_access_random>);
static_assert(
    sycl::is_property_of_v<sycl::ext::codeplay::usm_props::device_access_random,
                           sycl::usm_allocator<int, sycl::usm::alloc::host>>);

static_assert(
    sycl::is_property_v<sycl::ext::codeplay::usm_props::host_read_only>);
static_assert(
    sycl::is_property_of_v<sycl::ext::codeplay::usm_props::host_read_only,
                           sycl::usm_allocator<int, sycl::usm::alloc::host>>);

static_assert(
    sycl::is_property_v<sycl::ext::codeplay::usm_props::device_read_only>);
static_assert(
    sycl::is_property_of_v<sycl::ext::codeplay::usm_props::device_read_only,
                           sycl::usm_allocator<int, sycl::usm::alloc::host>>);

static_assert(
    sycl::is_property_v<sycl::ext::codeplay::usm_props::host_write_only>);
static_assert(
    sycl::is_property_of_v<sycl::ext::codeplay::usm_props::host_write_only,
                           sycl::usm_allocator<int, sycl::usm::alloc::host>>);

static_assert(
    sycl::is_property_v<sycl::ext::codeplay::usm_props::device_write_only>);
static_assert(
    sycl::is_property_of_v<sycl::ext::codeplay::usm_props::device_write_only,
                           sycl::usm_allocator<int, sycl::usm::alloc::host>>);

// check extension requirements from sycl_ext_usm_properties

static_assert(std::is_pod_v<sycl::ext::codeplay::usm_props::host_hot>);
static_assert(
    std::is_trivially_copyable_v<sycl::ext::codeplay::usm_props::host_hot>);
static_assert(
    std::is_nothrow_destructible_v<sycl::ext::codeplay::usm_props::host_hot>);
static_assert(std::is_nothrow_move_constructible_v<
              sycl::ext::codeplay::usm_props::host_hot>);

static_assert(std::is_pod_v<sycl::ext::codeplay::usm_props::device_hot>);
static_assert(
    std::is_trivially_copyable_v<sycl::ext::codeplay::usm_props::device_hot>);
static_assert(
    std::is_nothrow_destructible_v<sycl::ext::codeplay::usm_props::device_hot>);
static_assert(std::is_nothrow_move_constructible_v<
              sycl::ext::codeplay::usm_props::device_hot>);

static_assert(std::is_pod_v<sycl::ext::codeplay::usm_props::host_cold>);
static_assert(
    std::is_trivially_copyable_v<sycl::ext::codeplay::usm_props::host_cold>);
static_assert(
    std::is_nothrow_destructible_v<sycl::ext::codeplay::usm_props::host_cold>);
static_assert(std::is_nothrow_move_constructible_v<
              sycl::ext::codeplay::usm_props::host_cold>);

static_assert(std::is_pod_v<sycl::ext::codeplay::usm_props::device_cold>);
static_assert(
    std::is_trivially_copyable_v<sycl::ext::codeplay::usm_props::device_cold>);
static_assert(std::is_nothrow_destructible_v<
              sycl::ext::codeplay::usm_props::device_cold>);
static_assert(std::is_nothrow_move_constructible_v<
              sycl::ext::codeplay::usm_props::device_cold>);

static_assert(
    std::is_pod_v<sycl::ext::codeplay::usm_props::host_cache_non_coherent>);
static_assert(std::is_trivially_copyable_v<
              sycl::ext::codeplay::usm_props::host_cache_non_coherent>);
static_assert(std::is_nothrow_destructible_v<
              sycl::ext::codeplay::usm_props::host_cache_non_coherent>);
static_assert(std::is_nothrow_move_constructible_v<
              sycl::ext::codeplay::usm_props::host_cache_non_coherent>);

static_assert(
    std::is_pod_v<sycl::ext::codeplay::usm_props::device_cache_non_coherent>);
static_assert(std::is_trivially_copyable_v<
              sycl::ext::codeplay::usm_props::device_cache_non_coherent>);
static_assert(std::is_nothrow_destructible_v<
              sycl::ext::codeplay::usm_props::device_cache_non_coherent>);
static_assert(std::is_nothrow_move_constructible_v<
              sycl::ext::codeplay::usm_props::device_cache_non_coherent>);

static_assert(
    std::is_pod_v<sycl::ext::codeplay::usm_props::host_cache_write_combine>);
static_assert(std::is_trivially_copyable_v<
              sycl::ext::codeplay::usm_props::host_cache_write_combine>);
static_assert(std::is_nothrow_destructible_v<
              sycl::ext::codeplay::usm_props::host_cache_write_combine>);
static_assert(std::is_nothrow_move_constructible_v<
              sycl::ext::codeplay::usm_props::host_cache_write_combine>);

static_assert(
    std::is_pod_v<sycl::ext::codeplay::usm_props::device_cache_write_combine>);
static_assert(std::is_trivially_copyable_v<
              sycl::ext::codeplay::usm_props::device_cache_write_combine>);
static_assert(std::is_nothrow_destructible_v<
              sycl::ext::codeplay::usm_props::device_cache_write_combine>);
static_assert(std::is_nothrow_move_constructible_v<
              sycl::ext::codeplay::usm_props::device_cache_write_combine>);

static_assert(
    std::is_pod_v<sycl::ext::codeplay::usm_props::host_access_sequential>);
static_assert(std::is_trivially_copyable_v<
              sycl::ext::codeplay::usm_props::host_access_sequential>);
static_assert(std::is_nothrow_destructible_v<
              sycl::ext::codeplay::usm_props::host_access_sequential>);
static_assert(std::is_nothrow_move_constructible_v<
              sycl::ext::codeplay::usm_props::host_access_sequential>);

static_assert(
    std::is_pod_v<sycl::ext::codeplay::usm_props::device_access_sequential>);
static_assert(std::is_trivially_copyable_v<
              sycl::ext::codeplay::usm_props::device_access_sequential>);
static_assert(std::is_nothrow_destructible_v<
              sycl::ext::codeplay::usm_props::device_access_sequential>);
static_assert(std::is_nothrow_move_constructible_v<
              sycl::ext::codeplay::usm_props::device_access_sequential>);

static_assert(
    std::is_pod_v<sycl::ext::codeplay::usm_props::host_access_random>);
static_assert(std::is_trivially_copyable_v<
              sycl::ext::codeplay::usm_props::host_access_random>);
static_assert(std::is_nothrow_destructible_v<
              sycl::ext::codeplay::usm_props::host_access_random>);
static_assert(std::is_nothrow_move_constructible_v<
              sycl::ext::codeplay::usm_props::host_access_random>);

static_assert(
    std::is_pod_v<sycl::ext::codeplay::usm_props::device_access_random>);
static_assert(std::is_trivially_copyable_v<
              sycl::ext::codeplay::usm_props::device_access_random>);
static_assert(std::is_nothrow_destructible_v<
              sycl::ext::codeplay::usm_props::device_access_random>);
static_assert(std::is_nothrow_move_constructible_v<
              sycl::ext::codeplay::usm_props::device_access_random>);

static_assert(std::is_pod_v<sycl::ext::codeplay::usm_props::host_read_only>);
static_assert(std::is_trivially_copyable_v<
              sycl::ext::codeplay::usm_props::host_read_only>);
static_assert(std::is_nothrow_destructible_v<
              sycl::ext::codeplay::usm_props::host_read_only>);
static_assert(std::is_nothrow_move_constructible_v<
              sycl::ext::codeplay::usm_props::host_read_only>);

static_assert(std::is_pod_v<sycl::ext::codeplay::usm_props::device_read_only>);
static_assert(std::is_trivially_copyable_v<
              sycl::ext::codeplay::usm_props::device_read_only>);
static_assert(std::is_nothrow_destructible_v<
              sycl::ext::codeplay::usm_props::device_read_only>);
static_assert(std::is_nothrow_move_constructible_v<
              sycl::ext::codeplay::usm_props::device_read_only>);

static_assert(std::is_pod_v<sycl::ext::codeplay::usm_props::host_write_only>);
static_assert(std::is_trivially_copyable_v<
              sycl::ext::codeplay::usm_props::host_write_only>);
static_assert(std::is_nothrow_destructible_v<
              sycl::ext::codeplay::usm_props::host_write_only>);
static_assert(std::is_nothrow_move_constructible_v<
              sycl::ext::codeplay::usm_props::host_write_only>);

static_assert(std::is_pod_v<sycl::ext::codeplay::usm_props::device_write_only>);
static_assert(std::is_trivially_copyable_v<
              sycl::ext::codeplay::usm_props::device_write_only>);
static_assert(std::is_nothrow_destructible_v<
              sycl::ext::codeplay::usm_props::device_write_only>);
static_assert(std::is_nothrow_move_constructible_v<
              sycl::ext::codeplay::usm_props::device_write_only>);

void test_shared(sycl::queue &q) {
  int *p = nullptr;
  assert((p = sycl::malloc_shared<int>(
              N, q, {sycl::ext::codeplay::usm_props::host_hot()})));
  sycl::free(p, q);
  assert((p = sycl::malloc<int>(N, q,
                                {sycl::ext::codeplay::usm_props::host_hot()})));
  sycl::free(p, q);

  assert((p = sycl::malloc_shared<int>(
              N, q, {sycl::ext::codeplay::usm_props::device_hot()})));
  sycl::free(p, q);
  assert((p = sycl::malloc<int>(
              N, q, {sycl::ext::codeplay::usm_props::device_hot()})));
  sycl::free(p, q);

  assert((p = sycl::malloc_shared<int>(
              N, q, {sycl::ext::codeplay::usm_props::host_cold()})));
  sycl::free(p, q);
  assert((p = sycl::malloc<int>(
              N, q, {sycl::ext::codeplay::usm_props::host_cold()})));
  sycl::free(p, q);

  assert((p = sycl::malloc_shared<int>(
              N, q, {sycl::ext::codeplay::usm_props::device_cold()})));
  sycl::free(p, q);
  assert((p = sycl::malloc<int>(
              N, q, {sycl::ext::codeplay::usm_props::device_cold()})));
  sycl::free(p, q);

  assert(
      (p = sycl::malloc_shared<int>(
           N, q, {sycl::ext::codeplay::usm_props::host_cache_non_coherent()})));
  sycl::free(p, q);
  assert(
      (p = sycl::malloc<int>(
           N, q, {sycl::ext::codeplay::usm_props::host_cache_non_coherent()})));
  sycl::free(p, q);

  assert((p = sycl::malloc_shared<int>(
              N, q,
              {sycl::ext::codeplay::usm_props::device_cache_non_coherent()})));
  sycl::free(p, q);
  assert((p = sycl::malloc<int>(
              N, q,
              {sycl::ext::codeplay::usm_props::device_cache_non_coherent()})));
  sycl::free(p, q);

  assert((
      p = sycl::malloc_shared<int>(
          N, q, {sycl::ext::codeplay::usm_props::host_cache_write_combine()})));
  sycl::free(p, q);
  assert((
      p = sycl::malloc<int>(
          N, q, {sycl::ext::codeplay::usm_props::host_cache_write_combine()})));
  sycl::free(p, q);

  assert((p = sycl::malloc_shared<int>(
              N, q,
              {sycl::ext::codeplay::usm_props::device_cache_write_combine()})));
  sycl::free(p, q);
  assert((p = sycl::malloc<int>(
              N, q,
              {sycl::ext::codeplay::usm_props::device_cache_write_combine()})));
  sycl::free(p, q);

  assert(
      (p = sycl::malloc_shared<int>(
           N, q, {sycl::ext::codeplay::usm_props::host_access_sequential()})));
  sycl::free(p, q);
  assert(
      (p = sycl::malloc<int>(
           N, q, {sycl::ext::codeplay::usm_props::host_access_sequential()})));
  sycl::free(p, q);

  assert((
      p = sycl::malloc_shared<int>(
          N, q, {sycl::ext::codeplay::usm_props::device_access_sequential()})));
  sycl::free(p, q);
  assert((
      p = sycl::malloc<int>(
          N, q, {sycl::ext::codeplay::usm_props::device_access_sequential()})));
  sycl::free(p, q);

  assert((p = sycl::malloc_shared<int>(
              N, q, {sycl::ext::codeplay::usm_props::host_access_random()})));
  sycl::free(p, q);
  assert((p = sycl::malloc<int>(
              N, q, {sycl::ext::codeplay::usm_props::host_access_random()})));
  sycl::free(p, q);

  assert((p = sycl::malloc_shared<int>(
              N, q, {sycl::ext::codeplay::usm_props::device_access_random()})));
  sycl::free(p, q);
  assert((p = sycl::malloc<int>(
              N, q, {sycl::ext::codeplay::usm_props::device_access_random()})));
  sycl::free(p, q);

  assert((p = sycl::malloc_shared<int>(
              N, q, {sycl::ext::codeplay::usm_props::host_read_only()})));
  sycl::free(p, q);
  assert((p = sycl::malloc<int>(
              N, q, {sycl::ext::codeplay::usm_props::host_read_only()})));
  sycl::free(p, q);

  assert((p = sycl::malloc_shared<int>(
              N, q, {sycl::ext::codeplay::usm_props::device_read_only()})));
  sycl::free(p, q);
  assert((p = sycl::malloc<int>(
              N, q, {sycl::ext::codeplay::usm_props::device_read_only()})));
  sycl::free(p, q);

  assert((p = sycl::malloc_shared<int>(
              N, q, {sycl::ext::codeplay::usm_props::host_write_only()})));
  sycl::free(p, q);
  assert((p = sycl::malloc<int>(
              N, q, {sycl::ext::codeplay::usm_props::host_write_only()})));
  sycl::free(p, q);

  assert((p = sycl::malloc_shared<int>(
              N, q, {sycl::ext::codeplay::usm_props::device_write_only()})));
  sycl::free(p, q);
  assert((p = sycl::malloc<int>(
              N, q, {sycl::ext::codeplay::usm_props::device_write_only()})));
  sycl::free(p, q);
}

void test_host(sycl::queue &q) {
  int *p = nullptr;
  assert((p = sycl::malloc_host<int>(
              N, q, {sycl::ext::codeplay::usm_props::host_hot()})));
  sycl::free(p, q);
  assert((p = sycl::malloc<int>(N, q,
                                {sycl::ext::codeplay::usm_props::host_hot()})));
  sycl::free(p, q);

  assert((p = sycl::malloc_host<int>(
              N, q, {sycl::ext::codeplay::usm_props::device_hot()})));
  sycl::free(p, q);
  assert((p = sycl::malloc<int>(
              N, q, {sycl::ext::codeplay::usm_props::device_hot()})));
  sycl::free(p, q);

  assert((p = sycl::malloc_host<int>(
              N, q, {sycl::ext::codeplay::usm_props::host_cold()})));
  sycl::free(p, q);
  assert((p = sycl::malloc<int>(
              N, q, {sycl::ext::codeplay::usm_props::host_cold()})));
  sycl::free(p, q);

  assert((p = sycl::malloc_host<int>(
              N, q, {sycl::ext::codeplay::usm_props::device_cold()})));
  sycl::free(p, q);
  assert((p = sycl::malloc<int>(
              N, q, {sycl::ext::codeplay::usm_props::device_cold()})));
  sycl::free(p, q);

  assert(
      (p = sycl::malloc_host<int>(
           N, q, {sycl::ext::codeplay::usm_props::host_cache_non_coherent()})));
  sycl::free(p, q);
  assert(
      (p = sycl::malloc<int>(
           N, q, {sycl::ext::codeplay::usm_props::host_cache_non_coherent()})));
  sycl::free(p, q);

  assert((p = sycl::malloc_host<int>(
              N, q,
              {sycl::ext::codeplay::usm_props::device_cache_non_coherent()})));
  sycl::free(p, q);
  assert((p = sycl::malloc<int>(
              N, q,
              {sycl::ext::codeplay::usm_props::device_cache_non_coherent()})));
  sycl::free(p, q);

  assert((
      p = sycl::malloc_host<int>(
          N, q, {sycl::ext::codeplay::usm_props::host_cache_write_combine()})));
  sycl::free(p, q);
  assert((
      p = sycl::malloc<int>(
          N, q, {sycl::ext::codeplay::usm_props::host_cache_write_combine()})));
  sycl::free(p, q);

  assert((p = sycl::malloc_host<int>(
              N, q,
              {sycl::ext::codeplay::usm_props::device_cache_write_combine()})));
  sycl::free(p, q);
  assert((p = sycl::malloc<int>(
              N, q,
              {sycl::ext::codeplay::usm_props::device_cache_write_combine()})));
  sycl::free(p, q);

  assert(
      (p = sycl::malloc_host<int>(
           N, q, {sycl::ext::codeplay::usm_props::host_access_sequential()})));
  sycl::free(p, q);
  assert(
      (p = sycl::malloc<int>(
           N, q, {sycl::ext::codeplay::usm_props::host_access_sequential()})));
  sycl::free(p, q);

  assert((
      p = sycl::malloc_host<int>(
          N, q, {sycl::ext::codeplay::usm_props::device_access_sequential()})));
  sycl::free(p, q);
  assert((
      p = sycl::malloc<int>(
          N, q, {sycl::ext::codeplay::usm_props::device_access_sequential()})));
  sycl::free(p, q);

  assert((p = sycl::malloc_host<int>(
              N, q, {sycl::ext::codeplay::usm_props::host_access_random()})));
  sycl::free(p, q);
  assert((p = sycl::malloc<int>(
              N, q, {sycl::ext::codeplay::usm_props::host_access_random()})));
  sycl::free(p, q);

  assert((p = sycl::malloc_host<int>(
              N, q, {sycl::ext::codeplay::usm_props::device_access_random()})));
  sycl::free(p, q);
  assert((p = sycl::malloc<int>(
              N, q, {sycl::ext::codeplay::usm_props::device_access_random()})));
  sycl::free(p, q);

  assert((p = sycl::malloc_host<int>(
              N, q, {sycl::ext::codeplay::usm_props::host_read_only()})));
  sycl::free(p, q);
  assert((p = sycl::malloc<int>(
              N, q, {sycl::ext::codeplay::usm_props::host_read_only()})));
  sycl::free(p, q);

  assert((p = sycl::malloc_host<int>(
              N, q, {sycl::ext::codeplay::usm_props::device_read_only()})));
  sycl::free(p, q);
  assert((p = sycl::malloc<int>(
              N, q, {sycl::ext::codeplay::usm_props::device_read_only()})));
  sycl::free(p, q);

  assert((p = sycl::malloc_host<int>(
              N, q, {sycl::ext::codeplay::usm_props::host_write_only()})));
  sycl::free(p, q);
  assert((p = sycl::malloc<int>(
              N, q, {sycl::ext::codeplay::usm_props::host_write_only()})));
  sycl::free(p, q);

  assert((p = sycl::malloc_host<int>(
              N, q, {sycl::ext::codeplay::usm_props::device_write_only()})));
  sycl::free(p, q);
  assert((p = sycl::malloc<int>(
              N, q, {sycl::ext::codeplay::usm_props::device_write_only()})));
  sycl::free(p, q);
}

void test_device(sycl::queue &q) {
  int *p = nullptr;
  assert((p = sycl::malloc_device<int>(
              N, q, {sycl::ext::codeplay::usm_props::host_hot()})));
  sycl::free(p, q);
  assert((p = sycl::malloc<int>(N, q,
                                {sycl::ext::codeplay::usm_props::host_hot()})));
  sycl::free(p, q);

  assert((p = sycl::malloc_device<int>(
              N, q, {sycl::ext::codeplay::usm_props::device_hot()})));
  sycl::free(p, q);
  assert((p = sycl::malloc<int>(
              N, q, {sycl::ext::codeplay::usm_props::device_hot()})));
  sycl::free(p, q);

  assert((p = sycl::malloc_device<int>(
              N, q, {sycl::ext::codeplay::usm_props::host_cold()})));
  sycl::free(p, q);
  assert((p = sycl::malloc<int>(
              N, q, {sycl::ext::codeplay::usm_props::host_cold()})));
  sycl::free(p, q);

  assert((p = sycl::malloc_device<int>(
              N, q, {sycl::ext::codeplay::usm_props::device_cold()})));
  sycl::free(p, q);
  assert((p = sycl::malloc<int>(
              N, q, {sycl::ext::codeplay::usm_props::device_cold()})));
  sycl::free(p, q);

  assert(
      (p = sycl::malloc_device<int>(
           N, q, {sycl::ext::codeplay::usm_props::host_cache_non_coherent()})));
  sycl::free(p, q);
  assert(
      (p = sycl::malloc<int>(
           N, q, {sycl::ext::codeplay::usm_props::host_cache_non_coherent()})));
  sycl::free(p, q);

  assert((p = sycl::malloc_device<int>(
              N, q,
              {sycl::ext::codeplay::usm_props::device_cache_non_coherent()})));
  sycl::free(p, q);
  assert((p = sycl::malloc<int>(
              N, q,
              {sycl::ext::codeplay::usm_props::device_cache_non_coherent()})));
  sycl::free(p, q);

  assert((
      p = sycl::malloc_device<int>(
          N, q, {sycl::ext::codeplay::usm_props::host_cache_write_combine()})));
  sycl::free(p, q);
  assert((
      p = sycl::malloc<int>(
          N, q, {sycl::ext::codeplay::usm_props::host_cache_write_combine()})));
  sycl::free(p, q);

  assert((p = sycl::malloc_device<int>(
              N, q,
              {sycl::ext::codeplay::usm_props::device_cache_write_combine()})));
  sycl::free(p, q);
  assert((p = sycl::malloc<int>(
              N, q,
              {sycl::ext::codeplay::usm_props::device_cache_write_combine()})));
  sycl::free(p, q);

  assert(
      (p = sycl::malloc_device<int>(
           N, q, {sycl::ext::codeplay::usm_props::host_access_sequential()})));
  sycl::free(p, q);
  assert(
      (p = sycl::malloc<int>(
           N, q, {sycl::ext::codeplay::usm_props::host_access_sequential()})));
  sycl::free(p, q);

  assert((
      p = sycl::malloc_device<int>(
          N, q, {sycl::ext::codeplay::usm_props::device_access_sequential()})));
  sycl::free(p, q);
  assert((
      p = sycl::malloc<int>(
          N, q, {sycl::ext::codeplay::usm_props::device_access_sequential()})));
  sycl::free(p, q);

  assert((p = sycl::malloc_device<int>(
              N, q, {sycl::ext::codeplay::usm_props::host_access_random()})));
  sycl::free(p, q);
  assert((p = sycl::malloc<int>(
              N, q, {sycl::ext::codeplay::usm_props::host_access_random()})));
  sycl::free(p, q);

  assert((p = sycl::malloc_device<int>(
              N, q, {sycl::ext::codeplay::usm_props::device_access_random()})));
  sycl::free(p, q);
  assert((p = sycl::malloc<int>(
              N, q, {sycl::ext::codeplay::usm_props::device_access_random()})));
  sycl::free(p, q);

  assert((p = sycl::malloc_device<int>(
              N, q, {sycl::ext::codeplay::usm_props::host_read_only()})));
  sycl::free(p, q);
  assert((p = sycl::malloc<int>(
              N, q, {sycl::ext::codeplay::usm_props::host_read_only()})));
  sycl::free(p, q);

  assert((p = sycl::malloc_device<int>(
              N, q, {sycl::ext::codeplay::usm_props::device_read_only()})));
  sycl::free(p, q);
  assert((p = sycl::malloc<int>(
              N, q, {sycl::ext::codeplay::usm_props::device_read_only()})));
  sycl::free(p, q);

  assert((p = sycl::malloc_device<int>(
              N, q, {sycl::ext::codeplay::usm_props::host_write_only()})));
  sycl::free(p, q);
  assert((p = sycl::malloc<int>(
              N, q, {sycl::ext::codeplay::usm_props::host_write_only()})));
  sycl::free(p, q);

  assert((p = sycl::malloc_device<int>(
              N, q, {sycl::ext::codeplay::usm_props::device_write_only()})));
  sycl::free(p, q);
  assert((p = sycl::malloc<int>(
              N, q, {sycl::ext::codeplay::usm_props::device_write_only()})));
  sycl::free(p, q);
}

int main() {
  sycl::queue q;
  if (q.get_device().has(sycl::aspect::usm_shared_allocations))
    test_shared(q);
  if (q.get_device().has(sycl::aspect::usm_device_allocations))
    test_device(q);
  if (q.get_device().has(sycl::aspect::usm_host_allocations))
    test_host(q);
}
