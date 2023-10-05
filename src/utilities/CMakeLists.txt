target_include_directories(incflo PRIVATE ${CMAKE_CURRENT_LIST_DIR})

target_sources(incflo
   PRIVATE
   incflo_build_info.cpp
   incflo_steady_state.cpp
   io.cpp
   volWgtSum.H
   )
