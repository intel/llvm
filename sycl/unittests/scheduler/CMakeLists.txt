add_sycl_unittest(SchedulerTests OBJECT
    BlockedCommands.cpp
    FailedCommands.cpp
    FinishedCmdCleanup.cpp
    LeafLimit.cpp
    MemObjCommandCleanup.cpp
    CommandsWaitForEvents.cpp
    WaitAfterCleanup.cpp
    LinkedAllocaDependencies.cpp
    LeavesCollection.cpp
    NoUnifiedHostMemory.cpp
    StreamInitDependencyOnHost.cpp
    InOrderQueueDeps.cpp
    utils.cpp
)
