#pragma once

#include <vector>
#include <tuple>
#include <sycl/event.hpp>

namespace sycl {
inline namespace _V1 {

//#ifdef __DPCPP_ENABLE_UNFINISHED_KHR_EXTENSIONS
namespace khr {

template <typename... Requirements>
class requirements;

template <typename... Requirements>
void add_events(const requirements<Requirements...> &Reqs, std::vector<event> &Events);

template <typename... Requirements>
class requirements {
public:
  requirements(Requirements... r) : MRequirements(r...) {}

private:
  std::tuple<Requirements...> MRequirements;

	template <typename... R>
  friend void add_events(const requirements<R...> &Reqs, std::vector<event> &Events);
};

template <typename... Requirements>
requirements(Requirements... r) -> requirements<Requirements...>;

template <typename Requirement, typename T>
void add_requirement(std::vector<T> &ReqCont, const Requirement &Req) {
	if constexpr (std::is_same_v<Requirement, T>)
		ReqCont.push_back(Req);
}

template <typename Requirement, typename T, typename... Requirements>
void add_requirement(std::vector<T> &ReqCont, const Requirement &Req, const Requirements&... Rest) {
	if constexpr (std::is_same_v<Requirement, T>)
		ReqCont.push_back(Req);
	add_requirement(ReqCont, Rest...);
}

template <typename T, typename... Requirements, size_t... Is>
void add_requirements(const std::tuple<Requirements...> &ReqsTuple, std::vector<T> &ReqCont,
	std::index_sequence<Is...>) {
	add_requirement(ReqCont, std::get<Is>(ReqsTuple)...);
}

template <typename... Requirements>
void add_events(const sycl::khr::requirements<Requirements...> &Reqs, std::vector<event> &Events) {
	add_requirements(Reqs.MRequirements, Events, std::make_index_sequence<sizeof...(Requirements)>());
}

template <typename... Requirements>
constexpr bool has_events() {
	return (std::is_same_v<event, Requirements> || ...);
}

} // namespace khr
//#endif
} // namespace _V1
} // namespace sycl