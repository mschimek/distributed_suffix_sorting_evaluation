/*******************************************************************************
 * mpi/sort.hpp
 *
 * Copyright (C) 2018 Florian Kurpicz <florian.kurpicz@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <iterator>
#include <ostream>
#include <vector>

#include "atomic_sorting/sorting_wrapper.hpp"
#include "kamping/communicator.hpp"
#include "kamping/named_parameters.hpp"
#include "mpi/check_sort.hpp"
#include "mpi/environment.hpp"

#include <kamping/collectives/allgather.hpp>
#include <kamping/collectives/allreduce.hpp>
#include <kamping/collectives/alltoall.hpp>
#include <kamping/collectives/barrier.hpp>
#include <kamping/collectives/gather.hpp>
#include <kamping/communicator.hpp>
#include <kamping/p2p/isend.hpp>
#include <kamping/p2p/recv.hpp>

namespace dsss::mpi {

template <typename T>
struct Status {
  T first;
  T last;
  bool empty;
  friend std::ostream& operator<<(std::ostream& out, const Status& status) {
    return out << "(" << status.empty << ", " << status.first << ", "
               << status.last << ")" << std::endl;
  }
};

template <typename DataType, class Compare>
inline void shift(std::vector<DataType>& local_data, Compare comp) {
  using namespace kamping;
  Communicator comm;
  Status<DataType> status;
  status.empty = local_data.empty();
  if (!local_data.empty()) {
    status.first = local_data.front();
    status.last = local_data.back();
  }
  auto const recv_status = comm.allgather(send_buf(status));
  std::vector<int> send_counts(comm.size(), 0);
  std::vector<DataType> to_shift;
  if (comm.is_root()) {
    std::cout << "size: " << recv_status.size() << std::endl;
    for (const auto& elem : recv_status) {
      std::cout << elem << std::endl;
    }
  }
  comm.barrier();

  if (!local_data.empty()) {
    for (int i = comm.rank() - 1; i >= 0; --i) {
      auto const& cur_status = recv_status[i];
      if (!cur_status.empty) {
        if (comp(cur_status.last, local_data.front())) {
          // reached strictly smaller data
          break;
        }
        if ((!comp(cur_status.last, local_data.front()) &&
             comp(cur_status.first, local_data.front())) ||
            (!comp(cur_status.last, local_data.front()) && i == 0)) {
          if (local_data.size() > 0) {
            to_shift.push_back(local_data.front());
          }
          for (std::uint32_t j = 1; j < local_data.size(); ++j) {
            if (local_data[j] == local_data.front()) {
              to_shift.push_back(local_data[j]);
            } else {
              break;
            }
          }
          send_counts[i] = static_cast<int>(to_shift.size());
          local_data.erase(local_data.begin(),
                           local_data.begin() + to_shift.size());
          break;
          // send
        }
      }
    }
  }
  auto recv_data =
      comm.alltoallv(send_buf(to_shift), kamping::send_counts(send_counts));
  std::copy(recv_data.begin(), recv_data.end(), std::back_inserter(local_data));
}

template <typename DataType, class Compare>
inline void sort(std::vector<DataType>& local_data, Compare comp,
                 environment env = environment()) {
  kamping::Communicator comm;

  // size_t local_size = local_data.size();
  // size_t global_size = comm.allreduce_single(kamping::send_buf(local_size),
  // kamping::op(kamping::ops::plus<>{}));

  // if (global_size < comm.size() * 1024) {
  //   local_data = comm.gatherv(kamping::send_buf(local_data));
  //   if (comm.rank() != comm.root()) {
  //     local_data.clear();
  //   }
  //   std::sort(local_data.begin(), local_data.end(), comp);
  //   return;
  // }

  auto& sorter = get_sorter_instance();
  sorter.sort(local_data, comp);

  // struct LMInfo {
  //   size_t count = 0;
  //   DataType data = DataType();
  // };

  // RMInfo info = local_data.empty() ? RMInfo{} : RMInfo{1, local_data[0]};

  // for (int i = 1; i < local_data.size(); ++i) {
  //   if (local_data[i] == local_data[0]) {
  //     ++LMInfo.count;
  //   }
  // }

  // std::cout << comm.rank() << ": " << local_data.size() << std::endl;

  comm.barrier();
  if (comm.is_root()) {
    std::cout << "shift after sort" << std::endl;
  }
  shift(local_data, comp);

  // bool repeat = true;

  // while (repeat) {
  //   repeat = false;
  //   if (comm.rank() != 0) {
  //     std::vector<DataType> to_shift;
  //     if (local_data.size() > 0) {
  //       to_shift.push_back(local_data.front());
  //     }
  //     for (uint32_t i = 1; i < local_data.size(); ++i) {
  //       if (local_data[i] == local_data.front()) {
  //         to_shift.push_back(local_data[i]);
  //       } else {
  //         break;
  //       }
  //     }
  //     comm.isend(kamping::send_buf(to_shift),
  //                kamping::destination(comm.rank() - 1));

  //    if (!local_data.empty() && to_shift.size() == local_data.size()) {
  //      repeat = true;
  //    }

  //    local_data.erase(local_data.begin(),
  //                     local_data.begin() + to_shift.size());
  //  }

  //  comm.barrier();

  //  if (comm.rank() + 1 != comm.size()) {
  //    auto result = comm.recv<DataType>(kamping::source(comm.rank() + 1));
  //    std::copy(result.begin(), result.end(), std::back_inserter(local_data));
  //  }
  //  comm.barrier();

  //  repeat = comm.allreduce_single(kamping::send_buf(repeat),
  //                                 kamping::op(kamping::ops::logical_or<>{}));
  //}
  // if (comm.is_root()) {
  //  std::cout << "finished after sort" << std::endl;
  //}

  comm.barrier();
  if (comm.is_root()) {
    std::cout << "Sort completed" << std::endl;
  }
  comm.barrier();
  if (comm.is_root()) {
    std::cout << "distribution checked" << std::endl;
  }
}

template <typename DataType, class Compare, class SCompare>
inline void sort_(std::vector<DataType>& local_data, Compare comp,
                  SCompare scomp, environment env = environment()) {
  kamping::Communicator comm;

  // size_t local_size = local_data.size();
  // size_t global_size = comm.allreduce_single(kamping::send_buf(local_size),
  // kamping::op(kamping::ops::plus<>{}));

  // if (global_size < comm.size() * 1024) {
  //   local_data = comm.gatherv(kamping::send_buf(local_data));
  //   if (comm.rank() != comm.root()) {
  //     local_data.clear();
  //   }
  //   std::sort(local_data.begin(), local_data.end(), comp);
  //   return;
  // }

  auto& sorter = get_sorter_instance();
  sorter.sort(local_data, comp);

  // struct LMInfo {
  //   size_t count = 0;
  //   DataType data = DataType();
  // };

  // RMInfo info = local_data.empty() ? RMInfo{} : RMInfo{1, local_data[0]};

  // for (int i = 1; i < local_data.size(); ++i) {
  //   if (local_data[i] == local_data[0]) {
  //     ++LMInfo.count;
  //   }
  // }

  // std::cout << comm.rank() << ": " << local_data.size() << std::endl;

  comm.barrier();
  if (comm.is_root()) {
    std::cout << "shift after sort" << std::endl;
  }
  shift(local_data, scomp);

  // bool repeat = true;

  // while (repeat) {
  //   repeat = false;
  //   if (comm.rank() != 0) {
  //     std::vector<DataType> to_shift;
  //     if (local_data.size() > 0) {
  //       to_shift.push_back(local_data.front());
  //     }
  //     for (uint32_t i = 1; i < local_data.size(); ++i) {
  //       if (local_data[i] == local_data.front()) {
  //         to_shift.push_back(local_data[i]);
  //       } else {
  //         break;
  //       }
  //     }
  //     comm.isend(kamping::send_buf(to_shift),
  //                kamping::destination(comm.rank() - 1));

  //    if (!local_data.empty() && to_shift.size() == local_data.size()) {
  //      repeat = true;
  //    }

  //    local_data.erase(local_data.begin(),
  //                     local_data.begin() + to_shift.size());
  //  }

  //  comm.barrier();

  //  if (comm.rank() + 1 != comm.size()) {
  //    auto result = comm.recv<DataType>(kamping::source(comm.rank() + 1));
  //    std::copy(result.begin(), result.end(), std::back_inserter(local_data));
  //  }
  //  comm.barrier();

  //  repeat = comm.allreduce_single(kamping::send_buf(repeat),
  //                                 kamping::op(kamping::ops::logical_or<>{}));
  //}
  // if (comm.is_root()) {
  //  std::cout << "finished after sort" << std::endl;
  //}

  comm.barrier();
  if (comm.is_root()) {
    std::cout << "Sort completed" << std::endl;
  }
  comm.barrier();
  if (comm.is_root()) {
    std::cout << "distribution checked" << std::endl;
  }
}

template <typename DataType, class Compare>
inline void sort_ns(std::vector<DataType>& local_data, Compare comp,
                    environment env = environment()) {
  auto& sorter = get_sorter_instance();
  sorter.sort(local_data, comp);
}

}  // namespace dsss::mpi
