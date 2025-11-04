//==---------------- event_deps.hpp - SYCL event dependency utils ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/context_impl.hpp>
#include <detail/device_impl.hpp>
#include <detail/event_impl.hpp>
#include <detail/graph/graph_impl.hpp>
#include <detail/queue_impl.hpp>
#include <sycl/detail/cg_types.hpp>
#include <sycl/exception.hpp>

#include <vector>

namespace sycl {
inline namespace _V1 {
namespace detail {

/// Adds an event dependency to the list of dependencies, performing
/// a series of checks.
///
/// If the event is associated with a graph, and the queue is not,
/// the queue will be switched to a recording mode (transitive queue
/// recording feature).
///
/// The LockQueue template argument defines, whether the queue lock
/// should be acquired for the transition to a recording mode. It is
/// set to false in cases, where the event dependencies are set directly
/// in the command submission flow, where the lock is already aquired.
///
/// \param EventImpl Event to register as a dependency
/// \param EventsRegistered A list of already registered events, where
/// the event will be added.
/// \param QueueImpl A queue associated with the event dependencies. Can
/// be nullptr if no associated queue.
/// \param ContextImpl A context associated with a queue or graph.
/// \param DeviceImpl A device associated with a queue or graph.
/// \param GraphImpl A graph associated with a queue or a handler. Can
/// be nullptr if no associated graph.
/// \param CommandGroupType Type of command group.
template <bool LockQueue = true>
void registerEventDependency(
    const EventImplPtr &EventImpl, std::vector<EventImplPtr> &EventsRegistered,
    queue_impl *QueueImpl, const context_impl &ContextImpl,
    const device_impl &DeviceImpl,
    const ext::oneapi::experimental::detail::graph_impl *GraphImpl,
    CGType CommandGroupType) {

  if (!EventImpl)
    return;
  if (EventImpl->isDiscarded()) {
    throw sycl::exception(make_error_code(errc::invalid),
                          "Queue operation cannot depend on discarded event.");
  }

  // Async alloc calls adapter immediately. Any explicit/implicit dependencies
  // are handled at that point, including in order queue deps. Further calls to
  // depends_on after an async alloc are explicitly disallowed.
  if (CommandGroupType == CGType::AsyncAlloc) {
    throw sycl::exception(make_error_code(errc::invalid),
                          "Cannot submit a dependency after an asynchronous "
                          "allocation has already been executed!");
  }

  auto EventGraph = EventImpl->getCommandGraph();
  if (QueueImpl && EventGraph) {
    auto QueueGraph = QueueImpl->getCommandGraph();

    if (&EventGraph->getContextImpl() != &ContextImpl) {
      throw sycl::exception(
          make_error_code(errc::invalid),
          "Cannot submit to a queue with a dependency from a graph that is "
          "associated with a different context.");
    }

    if (&EventGraph->getDeviceImpl() != &DeviceImpl) {
      throw sycl::exception(
          make_error_code(errc::invalid),
          "Cannot submit to a queue with a dependency from a graph that is "
          "associated with a different device.");
    }

    if (QueueGraph && QueueGraph != EventGraph) {
      throw sycl::exception(sycl::make_error_code(errc::invalid),
                            "Cannot submit to a recording queue with a "
                            "dependency from a different graph.");
    }

    // If the event dependency has a graph, that means that the queue that
    // created it was in recording mode. If the current queue is not recording,
    // we need to set it to recording (implements the transitive queue recording
    // feature).
    if (!QueueGraph) {
      if constexpr (LockQueue) {
        EventGraph->beginRecording(*QueueImpl);
      } else {
        EventGraph->beginRecordingUnlockedQueue(*QueueImpl);
      }
    }
  }

  if (GraphImpl) {
    if (EventGraph == nullptr) {
      throw sycl::exception(
          make_error_code(errc::invalid),
          "Graph nodes cannot depend on events from outside the graph.");
    }
    if (EventGraph.get() != GraphImpl) {
      throw sycl::exception(
          make_error_code(errc::invalid),
          "Graph nodes cannot depend on events from another graph.");
    }
  }
  EventsRegistered.push_back(EventImpl);
}

} // namespace detail
} // namespace _V1
} // namespace sycl
