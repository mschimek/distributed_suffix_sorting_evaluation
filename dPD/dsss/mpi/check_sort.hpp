#pragma once

#include <iomanip>
#include <iostream>
#include <iterator>
#include <ostream>
#include <vector>

#include "atomic_sorting/sorting_wrapper.hpp"
#include "kamping/named_parameters.hpp"
#include "mpi/environment.hpp"

#include <kamping/collectives/allgather.hpp>
#include <kamping/collectives/allreduce.hpp>
#include <kamping/collectives/barrier.hpp>
#include <kamping/collectives/gather.hpp>
#include <kamping/communicator.hpp>
#include <kamping/p2p/isend.hpp>
#include <kamping/p2p/recv.hpp>

namespace dsss::mpi {

std::size_t inline check_sum(std::size_t local_elem, std::string const& msg,
                             std::size_t expected = -1) {
  using namespace kamping;
  Communicator comm;
  auto const distribution = comm.gatherv(send_buf(local_elem));
  auto const sum =
      comm.allreduce_single(send_buf(local_elem), op(std::plus<>{}));
  if (comm.is_root()) {
    if ((expected != -1) && (sum != expected)) {
      std::cout << "ERROR" << std::endl;
    }
    std::cout << msg << ":" << sum << std::endl;
    std::cout << "\t\t\t";
    bool has_zero = false;
    for (std::size_t i = 0; i < distribution.size(); ++i) {
      if (distribution[i] == 0) {
        has_zero = true;
      }
      std::cout << "i:" << i << std::setw(8) << distribution[i] << ",";
    }
    std::cout << std::endl;
    std::cout << "has zero: " << has_zero << std::endl;
  }
  return sum;
}

template <typename T>
struct Wrapper {
  bool not_empty;
  T first;
  T last;
};
template <typename DataType, class Compare>
inline void check_distribution(std::vector<DataType> const& local_data,
                               Compare comp) {
  using namespace kamping;
  kamping::Communicator comm;

  Wrapper<DataType> entry{false, DataType{}, DataType{}};
  if (!local_data.empty()) {
    auto& [not_empty, fi, la] = entry;
    not_empty = true;
    fi = local_data.front();
    la = local_data.back();
  }
  auto gathered_data = comm.allgather(send_buf(entry));
  bool correct = true;
  if (!local_data.empty()) {
    for (std::size_t i = comm.rank() + 1; i < comm.size(); ++i) {
      auto const& [not_empty, fi, la] = gathered_data[i];
      if (not_empty) {
        if (!comp(local_data.back(), fi)) {
          correct = false;
        }
        break;
      }
    }
  }
  correct = comm.allreduce_single(send_buf(correct), op(ops::logical_and<>{}));
  if (comm.rank() == 0 && !correct) {
    for (std::size_t i = 0; i < gathered_data.size(); ++i) {
      const auto& [empty, fir, la] = gathered_data[i];
      std::cout << i << " empty: " << empty << " " << fir << " " << la
                << std::endl;
    }
  }
}

template <typename DataType, class Compare>
inline void is_sorted(std::vector<DataType> const& local_data,
                               Compare comp) {
  using namespace kamping;
  kamping::Communicator comm;

  Wrapper<DataType> entry{false, DataType{}, DataType{}};
  if (!local_data.empty()) {
    auto& [not_empty, fi, la] = entry;
    not_empty = true;
    fi = local_data.front();
    la = local_data.back();
  }
  auto gathered_data = comm.allgather(send_buf(entry));
  const auto locally_sorted = std::is_sorted(local_data.begin(), local_data.end(), comp);
  if(!locally_sorted) {
    std::cout << "locally not sorted" << std::endl;
  }
  auto tmp = gathered_data;
  auto it = std::remove_if(tmp.begin(), tmp.end(), [&](const auto& elem) {
    return !elem.not_empty;
  });
  const auto globally_sorted = std::is_sorted(tmp.begin(), it, [&](const auto& lhs, const auto& rhs) {
    return comp(lhs.last, rhs.first);
  });
  bool correct = locally_sorted && globally_sorted;
  correct = comm.allreduce_single(send_buf(correct), op(ops::logical_and<>{}));
  if (comm.rank() == 0 && !correct) {
    std::cout << "ERROR not sorted" << std::endl;
    for (std::size_t i = 0; i < gathered_data.size(); ++i) {
      const auto& [empty, fir, la] = gathered_data[i];
      std::cout << i << " empty: " << empty << " " << fir << " " << la
                << std::endl;
    }
  }
}
}  // namespace dsss::mpi
