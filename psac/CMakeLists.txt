# This file has been modified by Manuel Haag, Matthias Schimek, 2025

cmake_minimum_required(VERSION 3.13)

# project settings
project(psac)

include(FetchContent)

FetchContent_Declare(
  kamping
  GIT_REPOSITORY https://github.com/kamping-site/kamping.git
  GIT_TAG 1effe84
  SYSTEM)

FetchContent_Declare(
  cli11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
  GIT_TAG v2.3.2
  SYSTEM)

FetchContent_Declare(magic-enum
  GIT_REPOSITORY  https://github.com/Neargye/magic_enum.git
  GIT_TAG         v0.9.7
  SYSTEM
)

FetchContent_Declare(tlx
  GIT_REPOSITORY  https://github.com/tlx/tlx.git
  GIT_TAG         cf83363
  SYSTEM
)

set(IPS4O_DISABLE_PARALLEL OFF)
FetchContent_Declare(ips4o
  GIT_REPOSITORY  https://github.com/mschimek/ips4o.git
  GIT_TAG         eb30352
  SYSTEM
)

set(MXX_BUILD_TESTS OFF CACHE BOOL "disable building mxx tests" FORCE)
FetchContent_Declare(mxx
  GIT_REPOSITORY  https://github.com/mschimek/mxx.git
  GIT_TAG         60bfeb9
  SYSTEM
)

FetchContent_MakeAvailable(kamping cli11 magic-enum tlx ips4o mxx)


FetchContent_Declare(kadis
  GIT_REPOSITORY  https://github.com/mschimek/KaDiS.git
  GIT_TAG         868a0e7
  SYSTEM
)
FetchContent_MakeAvailable(kadis)

# ams needs c++-14 feature 
set(CMAKE_CXX_STANDARD 20)
# added

#### additional Options
# compile options
option(USE_40_BIT_INDEX "use smallest data type possible for ranks in pdcx" OFF)
message(STATUS "---> USE_40_BIT_INDEX: ${USE_40_BIT_INDEX}")
#### additional Options
#### Options
OPTION(PSAC_BUILD_EXES "Build all psac executables (command line tools & benchmark scripts)" ON)
OPTION(PSAC_BUILD_TESTS "Build unit tests" ON)
OPTION(PSAC_ENABLE_COVERAGE "Enable code coverage reporting" OFF)

##### General Compilation Settings

# Initialize CXXFLAGS.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wuninitialized")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -flto=auto")

### Test Coverage
if(PSAC_ENABLE_COVERAGE)
  # turn off stack protection for gcov coverage, because the stack protector shows
  # up as a never taken branch, and thus turns any last statement in a function
  # with a stack procetor into a partially covered statement.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage -fno-stack-protector")
endif(PSAC_ENABLE_COVERAGE)

###### Executable and Libraries
# Save libs and executables in the same place
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

## psac header library
add_library(psaclib INTERFACE)
target_include_directories(psaclib INTERFACE include)

## psac dependencies

# added
#add_subdirectory(ext/KaDiS)
# added
# load mxx and its gtest wrapper
#set(MXX_BUILD_TESTS OFF CACHE BOOL "disable building mxx tests" FORCE)
#add_subdirectory(ext/mxx)
target_link_libraries(psaclib INTERFACE mxx)

# include libdivsufsort (with 64bit support but without examples)
set(BUILD_DIVSUFSORT64 ON CACHE BOOL "enables divsufsort 64bit functions" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "enables divsufsort examples" FORCE)
add_subdirectory(ext/libdivsufsort)

# prettyprint
target_include_directories(psaclib INTERFACE ext/cxx-prettyprint)

# divsufsort (integrate into psac as `psac-dss-lib`)
add_library(psac-dss-lib INTERFACE)
target_link_libraries(psac-dss-lib INTERFACE psaclib)
target_include_directories(psac-dss-lib INTERFACE ${libdivsufsort_BINARY_DIR}/include)
target_link_libraries(psac-dss-lib INTERFACE divsufsort divsufsort64)
target_link_libraries(psac-dss-lib INTERFACE tlx kamping kadis)


## build executables
if(PSAC_BUILD_EXES)
  add_subdirectory(src)
endif()

# build tests
if (PSAC_BUILD_TESTS)
  add_subdirectory(test)
endif()

