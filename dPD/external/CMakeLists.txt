################################################################################
# external/CMakeLists.txt
#
# Copyright (C) 2018 Florian Kurpicz <florian.kurpicz@tu-dortmund.de>
#
# All rights reserved. Published under the BSD-2 license in the LICENSE file.
################################################################################
# This file has been modified by Matthias Schimek, 2025

#add_subdirectory(kamping) # added
#add_subdirectory(KaDiS) # added
#add_subdirectory(googletest)
add_subdirectory(pss)

FetchContent_Declare(googletest
  GIT_REPOSITORY  https://github.com/google/googletest
  GIT_TAG         0bdccf4
  SYSTEM
)

FetchContent_MakeAvailable(googletest)
# Add TLX command line parser
add_library(dsss_tlx
  ${tlx_SOURCE_DIR}/tlx/cmdline_parser.cpp
  ${tlx_SOURCE_DIR}/tlx/digest/sha1.cpp
  ${tlx_SOURCE_DIR}/tlx/string/hexdump.cpp
  ${tlx_SOURCE_DIR}/tlx/string/parse_si_iec_units.cpp)

target_include_directories(dsss_tlx PUBLIC
  $<BUILD_INTERFACE:${tlx_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${tlx_SOURCE_DIR}>)

target_compile_options(dsss_tlx
  PRIVATE
  ${DSSS_FLAGS}
  $<$<CONFIG:Debug>:${DSSS_DEBUG_FLAGS}>
  $<$<CONFIG:Release>:${DSSS_RELEASE_FLAGS}>)

#add_library(ips4o INTERFACE)
#target_include_directories(ips4o INTERFACE ips4o/ips4o/)

################################################################################
