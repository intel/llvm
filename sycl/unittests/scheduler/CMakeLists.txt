set(clang $<TARGET_FILE:clang>)

set(__cxx_comp_backup ${CMAKE_CXX_COMPILER})
set(__c_comp_backup ${CMAKE_C_COMPILER})

set(CMAKE_C_COMPILER ${clang})
set(CMAKE_CXX_COMPILER ${clang})

add_sycl_unittest(SchedulerTests
    BlockedCommands.cpp
    FinishedCmdCleanup.cpp
    LeafLimit.cpp
    MemObjCommandCleanup.cpp
    utils.cpp
)

set(CMAKE_C_COMPILER ${__c_comp_backup})
set(CMAKE_CXX_COMPILER ${__cxx_comp_backup})
