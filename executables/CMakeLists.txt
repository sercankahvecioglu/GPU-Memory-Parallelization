add_executable(main main.cpp)
target_link_libraries(main PRIVATE lib)

add_executable(milestone4 milestone4.cpp)
target_link_libraries(milestone4 PRIVATE lib)

add_executable(milestone5 milestone5.cpp)
target_link_libraries(milestone5 PRIVATE lib)

add_executable(milestone6 milestone6.cpp)
target_link_libraries(milestone6 PRIVATE lib)

# Communication checks and distributed-solver regression tests.
foreach(ranks IN ITEMS 1 2 3)
    add_test(NAME Milestone6.Halos${ranks}Ranks
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:milestone6> ${MPIEXEC_POSTFLAGS}
                7 5 --check-halos)
    set_tests_properties(Milestone6.Halos${ranks}Ranks PROPERTIES
        PROCESSORS ${ranks} TIMEOUT 30)
endforeach()
add_test(NAME Milestone6.SingleColumnSingleRow
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 3
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:milestone6> ${MPIEXEC_POSTFLAGS}
            3 1 --check-halos)
set_tests_properties(Milestone6.SingleColumnSingleRow PROPERTIES PROCESSORS 3 TIMEOUT 30)

foreach(ranks IN ITEMS 1 2 3)
    foreach(steps IN ITEMS 0 1 200)
        add_test(NAME Milestone6.Solver${ranks}Ranks${steps}Steps
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
                    ${MPIEXEC_PREFLAGS} $<TARGET_FILE:milestone6> ${MPIEXEC_POSTFLAGS}
                    17 13 --steps ${steps} --check-solver)
        set_tests_properties(Milestone6.Solver${ranks}Ranks${steps}Steps PROPERTIES
            PROCESSORS ${ranks} TIMEOUT 30)
    endforeach()
endforeach()
add_test(NAME Milestone6.SolverSingleColumn
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 3
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:milestone6> ${MPIEXEC_POSTFLAGS}
            3 5 --steps 25 --check-solver)
set_tests_properties(Milestone6.SolverSingleColumn PROPERTIES PROCESSORS 3 TIMEOUT 30)
