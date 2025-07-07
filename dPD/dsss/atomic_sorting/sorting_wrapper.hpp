#pragma once

#include <mpi.h>

#include <random>
#include <string>
#include <vector>

#include "AmsSort/AmsSort.hpp"
#include "Bitonic/Bitonic.hpp"
#include "RBC.hpp"
#include "RFis/RFis.hpp"
#include "RQuick/RQuick.hpp"
#include "atomic_sorting/sample_sort.hpp"
#include "mpi/type_mapper.hpp"

namespace dsss::mpi {

enum AtomicSorters { SampleSort, Rquick, Ams, Bitonic, RFis };
static std::vector<std::string> atomic_sorter_names = {
    "sample_sort", "rquick", "ams", "bitonic", "rfis"};

struct SortingWrapper {
  SortingWrapper(environment env = environment())
      : mpi_comm(env.communicator()),
        data_seed(3469931 + env.rank()),
        gen(data_seed),
        num_levels(1),
        tag(123),
        sorter(AtomicSorters::SampleSort) {
  }

  void set_sorter(AtomicSorters new_sorter) { sorter = new_sorter; }
  void set_num_levels(int new_num_levels) { num_levels = new_num_levels; }
  void finalize_setting() {
    if (sorter == AtomicSorters::Ams && num_levels == 1) {
      is_init = true;
      RBC::Create_Comm_from_MPI(mpi_comm, &rcomm, true, true, true);
    } else {
      is_init = true;
      RBC::Create_Comm_from_MPI(mpi_comm, &rcomm);
    }
  }

  template <typename DataType, class Compare>
  inline void sort(std::vector<DataType>& local_data, Compare comp) {
    data_type_mapper<DataType> dtm;
    MPI_Datatype my_mpi_type = dtm.get_mpi_type();
    switch (sorter) {
      case SampleSort:
        // std::cout << "sample sort \n";
        sample_sort(local_data, comp);
        break;
      case Rquick:
        RQuick::sort(my_mpi_type, local_data, tag, gen, mpi_comm, comp);
        break;
      case Ams:
        if (!is_init) {
          std::cout << "is init: " << is_init << std::endl;
        }
        Ams::sortLevel(my_mpi_type, local_data, num_levels, gen, rcomm, comp);
        break;
      case Bitonic:
        Bitonic::Sort(local_data, my_mpi_type, tag, rcomm, comp);
        break;
      case RFis:
        RFis::Sort(my_mpi_type, local_data, rcomm, comp);
        break;
      default:
        sample_sort(local_data, comp);
    }
  }

  MPI_Comm mpi_comm;
  RBC::Comm rcomm;

  int data_seed;
  std::mt19937_64 gen;
  int num_levels;
  int tag;
  bool is_init = false;

  AtomicSorters sorter;
};

// singleton instance
inline SortingWrapper& get_sorter_instance() {
  static SortingWrapper wrapper;
  return wrapper;
}
}  // namespace dsss::mpi
