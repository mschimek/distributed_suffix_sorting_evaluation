// This file has been modified by Manuel Haag, Matthias Schimek, 2025

#ifndef IDX_SORT_HPP
#define IDX_SORT_HPP

#include <vector>
#include <mxx/comm.hpp>
#include <mxx/sort.hpp>
#include <mxx/datatypes.hpp>

// added
#include "print.hpp"
#include "sorting_wrapper.hpp"
#include "uint_types.hpp"


template <typename T, typename index_t>
struct TwoVecIdx {
    T v1;
    T v2;
    index_t idx;
};

namespace mxx {
template <typename T, typename index_t>
MXX_CUSTOM_TEMPLATE_STRUCT(MXX_WRAP_TEMPLATE(TwoVecIdx<T,index_t>), v1, v2, idx);
} // namespace mxx


template <typename T, typename index_t, bool Stable = false>
std::vector<index_t> idxsort_vectors(std::vector<T>& vec1, std::vector<T>& vec2, const mxx::comm& comm) {
    MXX_ASSERT(vec1.size() == vec2.size());
    //SAC_TIMER_START();

    size_t local_size = vec1.size();
    size_t prefix = mxx::exscan(local_size, comm);
    // convert the struct of arrays (local_SA, local_B, etc) into
    // array of structs (TwoBSA {.B1, .B2, .SA}) for sorting purposes

    // initialize tuple array
    using temp_index = index_t;
    std::vector<TwoVecIdx<temp_index, temp_index> > tuple_vec(local_size);

    // fill tuple vector
    for (std::size_t i = 0; i < local_size; ++i) {
        tuple_vec[i].v1 = temp_index(vec1[i]);
        tuple_vec[i].v2 = temp_index(vec2[i]);
        assert(prefix + i < std::numeric_limits<temp_index>::max());
        tuple_vec[i].idx = temp_index(prefix + i);
    }

    // code before
    // for (std::size_t i = 0; i < local_size; ++i) {
    //     tuple_vec[i].v1 = vec1[i];
    //     tuple_vec[i].v2 = vec2[i];
    //     assert(prefix + i < std::numeric_limits<index_t>::max());
    //     tuple_vec[i].idx = prefix + i;
    // }

    // release memory of input (to remain at the minimum 6x words memory usage)
    vec1 = std::vector<T>();
    vec2 = std::vector<T>();

    //SAC_TIMER_END_SECTION("isa2sa_tupleize");

    // using TT = TwoVecIdx<T, index_t>;
    using TT = TwoVecIdx<temp_index, temp_index>;
    auto cmp = [](const TT& x, const TT& y) {
        return x.v1 < y.v1 || (x.v1 == y.v1 && x.v2 < y.v2);
    };
    auto cmpidx = [](const TT& x, const TT& y) {
        return x.v1 < y.v1 || (x.v1 == y.v1 && x.v2 < y.v2)
               || (x.v1 == y.v1 && x.v2 == y.v2 && x.idx < y.idx);
    };

    auto& sorter = dsss::mpi::get_sorting_instance(comm);
    // parallel, distributed sample-sorting of tuples (B1, B2, SA)
    if (Stable) {
        sorter.sort(tuple_vec, cmpidx);
        //mxx::sort(tuple_vec.begin(), tuple_vec.end(), cmpidx, comm);
    }
    else {
        sorter.sort(tuple_vec, cmp);
        //mxx::sort(tuple_vec.begin(), tuple_vec.end(), cmp, comm);
    }


    //SAC_TIMER_END_SECTION("isa2sa_samplesort");

    // reallocate output
    vec1.resize(local_size);
    vec2.resize(local_size);
    std::vector<index_t> idx(local_size);

    // code before
    // back-convert array of structs into struct of arrays

    // read back into input vectors
    //for (std::size_t i = 0; i < local_size; ++i) {
    //    vec1[i] = tuple_vec[i].v1;
    //    vec2[i] = tuple_vec[i].v2;
    //    idx[i] = tuple_vec[i].idx;
    //}
    // 3 x 8 bytes + 3 x 5 bytes
    for (std::size_t i = 0; i < local_size; ++i) {
      vec1[i] = static_cast<T>(tuple_vec[i].v1);
      vec2[i] = static_cast<T>(tuple_vec[i].v2);
      idx[i] = index_t(tuple_vec[i].idx);
    }
    //SAC_TIMER_END_SECTION("isa2sa_untupleize");

    return idx;
}


#endif // IDX_SORT_HPP
