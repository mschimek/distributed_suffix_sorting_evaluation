// This file has been added by Manuel Haag, Matthias Schimek, 2025

#pragma once

#include <mpi.h>

#include <cstdint>
#include <mxx/comm.hpp>
#include <mxx/sort.hpp>
#include <random>
#include <vector>

#include "AmsSort/AmsSort.hpp"
#include "RBC.hpp"
#include "distribute.hpp"
#include "ips4o.hpp"
#include "kamping/communicator.hpp"
#include "kamping/mpi_datatype.hpp"

namespace dsss::mpi {

struct SortingWrapper {
  SortingWrapper(const mxx::comm& _comm)
      : comm(_comm),
        mpi_comm(_comm),
        kamping_comm(_comm),
        data_seed(3469931 + comm.rank()),
        gen{data_seed},
        tag(123),
        num_levels(1),
        use_ams(false) {}

  SortingWrapper(const SortingWrapper&) = delete;
  SortingWrapper& operator=(const SortingWrapper&) = delete;
  SortingWrapper& operator=(SortingWrapper&&) = delete;

  void set_num_levels(int new_num_levels) { num_levels = new_num_levels; }
  void set_use_ams(bool _use_ams) { use_ams = _use_ams; }
  void finalize_setting() {
    if (use_ams && num_levels == 1) {
      is_init = true;
      RBC::Create_Comm_from_MPI(mpi_comm, &rcomm, true, true, true);
    } else {
      is_init = true;
      RBC::Create_Comm_from_MPI(mpi_comm, &rcomm);
    }
  }

  template <typename DataType, class Compare>
  inline void sort_with_ams(std::vector<DataType>& local_data, Compare comp) {
    uint64_t size_before = local_data.size();
    MPI_Datatype my_mpi_type = kamping::mpi_datatype<DataType>();
    Ams::sortLevel(my_mpi_type, local_data, num_levels, gen, rcomm, comp);

    // need to redistribute data, because mxx::sort does so and algorithm
    // excepts this behavior
    local_data = dsss::mpi_util::distribute_data_custom(local_data, size_before, kamping_comm);
  }
  template <typename DataType, class Compare>
  inline void sort(std::vector<DataType>& local_data, Compare comp) {
    if (use_ams) {
      if (!is_init) {
        std::cout << "is init: " << is_init << std::endl;
      }
      sort_with_ams(local_data, comp);
    } else {
      mxx::sort(local_data.begin(), local_data.end(), comp, comm);
    }
  }

  const mxx::comm& comm;
  MPI_Comm mpi_comm;
  kamping::Communicator<> kamping_comm;

  RBC::Comm rcomm;

  int data_seed;
  std::mt19937_64 gen;
  int tag;

  int num_levels;
  bool use_ams;
  bool use_ips4o;
  bool is_init = false;
};

// singleton instance
inline SortingWrapper& get_sorting_instance(const mxx::comm& comm) {
  static SortingWrapper stats(comm);
  return stats;
}

}  // namespace dsss::mpi
namespace dsss {
struct LocalSortingWrapper {
  LocalSortingWrapper() : use_ips4o(false) {}

  void set_use_ips4o(bool _use_ips4o) { use_ips4o = _use_ips4o; }

  template <typename DataType, typename Compare>
  inline void local_sort(std::vector<DataType>& data, Compare&& comp) {
    if (use_ips4o) {
      ips4o::sort(data.begin(), data.end(), std::forward<Compare>(comp));
    } else {
      std::sort(data.begin(), data.end(), std::forward<Compare>(comp));
    }
  }
  bool use_ips4o;
};
// singleton instance
inline LocalSortingWrapper& get_local_sorting_instance() {
  static LocalSortingWrapper stats;
  return stats;
}
}  // namespace dsss
