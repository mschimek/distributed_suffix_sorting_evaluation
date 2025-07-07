################################################################################
# CMakeLists.txt
#
# Copyright (C) 2018 Florian Kurpicz <florian.kurpicz@tu-dortmund.de>
#
# All rights reserved. Published under the BSD-2 license in the LICENSE file.
################################################################################

cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/misc/cmake")

project(dsss)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif(CCACHE_FOUND)

option(DSSS_OVERSUBSCRIBE
  "Run tests with mpirun [...] --oversubscribe.
  This may be necessary for environments with less than four cores." OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif(NOT CMAKE_BUILD_TYPE)

# TODO: Look for a "modern" (target based) working way
find_package(MPI REQUIRED)
if (MPI_FOUND)
  message(STATUS "MPI FOUND: ${MPI_C_INCLUDE_PATH}")
  include_directories(${MPI_INCLUDE_PATH})
endif()

set(DSSS_FLAGS "-Wall;-pedantic;-Wextra")
set(DSSS_DEBUG_FLAGS "-O0;-ggdb")
set(DSSS_RELEASE_FLAGS "-O3;-march=native;-DNDEBUG")

include(FetchContent)

FetchContent_Declare(
  kamping
  GIT_REPOSITORY https://github.com/kamping-site/kamping.git
  GIT_TAG 1effe84
  SYSTEM)

FetchContent_Declare(tlx
  GIT_REPOSITORY  https://github.com/tlx/tlx.git
  GIT_TAG         cf83363
  SYSTEM
)

FetchContent_MakeAvailable(kamping tlx)

set(IPS4O_DISABLE_PARALLEL OFF)
FetchContent_Declare(ips4o
  GIT_REPOSITORY  https://github.com/mschimek/ips4o.git
  GIT_TAG         eb30352
  SYSTEM
)

FetchContent_MakeAvailable(ips4o)

FetchContent_Declare(kadis
  GIT_REPOSITORY  https://github.com/mschimek/KaDiS.git
  GIT_TAG         868a0e7
  SYSTEM
)
FetchContent_MakeAvailable(kadis)


add_subdirectory(dsss)
add_subdirectory(benchmark)
add_subdirectory(external)
add_subdirectory(tests)

message(STATUS "Built Type: " ${CMAKE_BUILD_TYPE} )

################################################################################
