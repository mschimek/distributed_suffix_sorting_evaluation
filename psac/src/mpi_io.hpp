/*******************************************************************************
 * mpi/distribute_input.hpp
 *
 * Copyright (C) 2018 Florian Kurpicz <florian.kurpicz@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
 ******************************************************************************/
// This file has been added by Manuel Haag, Matthias Schimek, 2025


#pragma once

#include <mpi.h>
#include <sys/types.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace mpi {

static std::string distribute_string(mxx::comm& comm,
                                     const std::string& input_path,
                                     size_t max_size = 0) {
  using char_type = uint8_t;
  MPI_File mpi_file;
  //   MPI_Comm mpi_comm = comm();

  //   MPI_File_open(mpi_comm,
  MPI_File_open(comm,
                (char*)input_path.c_str(),  // ugly cast to use old C interface
                MPI_MODE_RDONLY, MPI_INFO_NULL, &mpi_file);

  MPI_Offset global_file_size = 0;
  MPI_File_get_size(mpi_file, &global_file_size);
  if (max_size > 0) {
    global_file_size =
        std::min(max_size, static_cast<size_t>(global_file_size));
  }

  size_t local_slice_size = global_file_size / comm.size();
  int64_t larger_slices = global_file_size % comm.size();

  size_t offset;
  if (comm.rank() < larger_slices) {
    ++local_slice_size;
    offset = local_slice_size * comm.rank();
  } else {
    offset = larger_slices * (local_slice_size + 1);
    offset += (comm.rank() - larger_slices) * local_slice_size;
  }

  MPI_File_seek(mpi_file, offset, MPI_SEEK_SET);

  std::vector<char_type> result(local_slice_size);

  MPI_File_read(mpi_file, result.data(), local_slice_size, MPI_BYTE,
                MPI_STATUS_IGNORE);

  std::string str(result.begin(), result.end());
  return str;
}

}  // namespace mpi
