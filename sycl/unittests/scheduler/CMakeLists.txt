add_sycl_unittest(SchedulerTests OBJECT
    BlockedCommands.cpp
    Commands.cpp
    FailedCommands.cpp
    LeafLimit.cpp
    MemObjCommandCleanup.cpp
    CommandsWaitForEvents.cpp
    LinkedAllocaDependencies.cpp
    LeavesCollection.cpp
    NoHostUnifiedMemory.cpp
    StreamInitDependencyOnHost.cpp
    InOrderQueueDeps.cpp
    InOrderQueueHostTaskDeps.cpp
    AllocaLinking.cpp
    RequiredWGSize.cpp
    QueueFlushing.cpp
    GraphCleanup.cpp
    utils.cpp
    LeafLimitDiffContexts.cpp
    InOrderQueueSyncCheck.cpp
    EnqueueWithDependsOnDeps.cpp
    AccessorDefaultCtor.cpp
    HostTaskAndBarrier.cpp
    BarrierDependencies.cpp
)

if (SYCL_ENABLE_XPTI_TRACING)
  target_compile_definitions(SchedulerTests
    PRIVATE XPTI_ENABLE_INSTRUMENTATION XPTI_STATIC_LIBRARY)
endif()
