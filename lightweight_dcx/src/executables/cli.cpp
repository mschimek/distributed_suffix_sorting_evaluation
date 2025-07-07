
#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>

#include <CLI/App.hpp>
#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>
#include <kamping/communicator.hpp>
#include <kamping/measurements/counter.hpp>
#include <kamping/measurements/printer.hpp>
#include <kamping/named_parameters.hpp>
#include <magic_enum/magic_enum.hpp>
#include <tlx/cmdline_parser.hpp>

#include "CLI_mpi.hpp"
#include "mpi/io.hpp"
#include "mpi/reduce.hpp"
#include "options.hpp"
#include "pdcx/config.hpp"
#include "pdcx/difference_cover.hpp"
#include "pdcx/pdcx.hpp"
#include "sa_check.hpp"
#include "sorters/sample_sort_config.hpp"
#include "sorters/seq_string_sorter_wrapper.hpp"
#include "sorters/sorting_wrapper.hpp"
#include "strings/char_container.hpp"
#include "util/memory.hpp"
#include "util/printing.hpp"
#include "util/random.hpp"
#include "util/uint_types.hpp"

#define V(x) std::string(#x "=") << (x) << " " //"x=...

using namespace dsss;

inline void print_as_jsonlist_to_file(std::vector<std::string> objects, std::string filename) {
    std::ofstream outstream(filename);
    outstream << "[" << std::endl;
    for (std::size_t i = 0; i < objects.size(); ++i) {
        if (i > 0) {
            outstream << "," << std::endl;
        }
        outstream << objects[i];
    }
    outstream << std::endl << "]" << std::endl;
}

template <typename Enum>
auto string_map() -> std::map<std::string, Enum> {
    std::map<std::string, Enum> result;
    auto entries = magic_enum::enum_entries<Enum>();
    for (auto const& entry: entries) {
        result.emplace(entry.second, entry.first);
    }
    return result;
}


struct Parameters {
    size_t textsize = 0u;
    size_t alphabet_size = 2u;
    size_t external_iteration = 0;
    std::string algorithm = "DCX";
    std::string input_path = "";
    std::string output_path = "";
    std::string json_output_path = "";
    std::string dcx_variant = "dc3";
    std::size_t num_pe = 0u;
    bool check = false;
    dcx::PDCXConfig pdcx_config;


    std::vector<std::pair<std::string, std::string>> config() const {
        std::vector<std::pair<std::string, std::string>> config_vector;
        config_vector.emplace_back("textsize", std::to_string(textsize));
        config_vector.emplace_back("alphabet_size", std::to_string(alphabet_size));
        config_vector.emplace_back("num_pe", std::to_string(num_pe));
        config_vector.emplace_back("input_path", input_path);
        config_vector.emplace_back("output_path", output_path);
        config_vector.emplace_back("json_output_path", json_output_path);
        config_vector.emplace_back("dcx_variant", dcx_variant);
        config_vector.emplace_back("algorithm", algorithm);
        config_vector.emplace_back("external_iteration", std::to_string(external_iteration));
        {
            auto pdcx_confi_vector = pdcx_config.config();
            config_vector.insert(config_vector.end(),
                                 pdcx_confi_vector.begin(),
                                 pdcx_confi_vector.end());
        }

        return config_vector;
    }

    friend std::ostream& operator<<(std::ostream& out, Parameters const& params) {
        for (auto const& config_entry: params.config()) {
            out << config_entry.first << "=" << config_entry.second << " ";
        }
        return out;
    }
};

using char_type = uint8_t;
// using char_type = uint16_t;
// using char_type = uint32_t;
using index_type = uint40;

size_t string_size = {0};
size_t alphabet_size = {2};
size_t seed = {0};
std::string input_path = "";
std::string output_path = "";
std::string json_output_path = "";
std::string dcx_variant = "dc3";
bool check = false;

// dcx::PDCXConfig pdcx_config;

std::string atomic_sorter = "SampleSort";
std::string string_sorter = "RadixSortCI3";

std::string buckets_samples;
std::string buckets_merging;

dsss::mpi::SampleSortConfig sample_sort_config;
std::string splitter_sampling = "Uniform";
std::string splitter_sorting = "Central";

tlx::CmdlineParser cp;
std::vector<char_type> local_string;
std::vector<index_type> local_sa;

uint64_t input_alphabet_size = 0;

void check_limit(std::vector<uint32_t>& vec,
                 uint64_t limit,
                 std::string name,
                 kamping::Communicator<> const& comm = kamping::comm_world()) {
    if (vec.size() == 0)
        return;
    uint32_t _max = *std::max_element(vec.begin(), vec.end());
    if (_max > limit) {
        kamping::report_on_root(name + " must be <= " + std::to_string(limit) + ".", comm);
        exit(1);
    }
}

Parameters read_cli_parameters(int argc, char const** argv) {
    Parameters parameters;
    CLI::App app{"Suffix Sorting Benchmark"};
    app.add_option("--input",
                   parameters.input_path,
                   "Path to input file. The special input 'random' generates a random text of the "
                   "size given by parameter '-s'.");
    app.add_option("--textsize",
                   parameters.textsize,
                   "Size (in bytes unless stated otherwise) of the string that use to test our "
                   "suffix array construction algorithms.");
    app.add_option("--iteration",
                   parameters.external_iteration,
                   "Helper Argument for benchmarking.");
    app.add_option("--alphabet_size",
                   parameters.alphabet_size,
                   "Size of the alphbet used for random.");
    app.add_option("--output",
                   parameters.output_path,
                   "Filename for the output (SA). Note that the output is five times larger than "
                   "the input file.");
    app.add_option("--json-output-path", parameters.json_output_path, "path to json output");

    // pdcx configuration
    app.add_option("--seed",
                   parameters.pdcx_config.seed,
                   "Seed to be used for random. PE i uses seed: seed + i");
    app.add_flag("--check", parameters.check, "Check if the SA has been constructed correctly.");
    app.add_option("--dcx",
                   parameters.dcx_variant,
                   "DCX variant to use. Available options: dc3, dc7, dc13, ..., dc133.");
    app.add_option("--discarding-threshold",
                   parameters.pdcx_config.discarding_threshold,
                   "Value between [0, 1], threshold when to use discarding optimization.");
    app.add_option("--num-samples-splitters",
                   parameters.pdcx_config.num_samples_splitters,
                   "Total number of random samples to use to determine bucket splitters in space "
                   "efficient sort.");
    app.add_flag("--use-random-sampling-splitters",
                 parameters.pdcx_config.use_random_sampling_splitters,
                 "Use random sampling to determine block splitters in space efficient sort. ");
    app.add_flag("--balance-blocks-space-efficient-sort",
                 parameters.pdcx_config.balance_blocks_space_efficient_sort,
                 "Balance blocks after materialization in space efficient sorting.");
    uint64_t const num_buckets_limit = std::numeric_limits<uint16_t>::max();
    app.add_option("--buckets-sample-phase",
                   parameters.pdcx_config.buckets_samples,
                   "Number of buckets to use for space efficient sorting of samples on each "
                   "recursion level. Missing values default to 1. Example: 16,8,4")
        ->delimiter(',')
        ->check(CLI::Range(num_buckets_limit));
    app.add_option("--buckets-merging-phase",
                   parameters.pdcx_config.buckets_merging,
                   "Number of buckets to use for space efficient sorting in merging phase on each "
                   "recursion level. Missing values default to 1. Example: 16,8,4. If you "
                   "use large bucket sizes you should also set num_samples_splitters (-m) high "
                   "enough. 16 b log b for b buckets should be enough.")
        ->delimiter(',')
        ->check(CLI::Range(num_buckets_limit));
    app.add_option("--buckets-phase3",
                   parameters.pdcx_config.buckets_phase3,
                   "Number of buckets to use for space efficient sorting in Phase 3 for rri.");
    app.add_option("--samples-buckets-phase3", parameters.pdcx_config.num_samples_phase3);
    app.add_flag("--use-randomized-chunks",
                 parameters.pdcx_config.use_randomized_chunks,
                 "Use randomized chunks in bucket sorting to distribute work.");
    app.add_option("--avg-chunks-pe",
                   parameters.pdcx_config.avg_chunks_pe,
                   "Average number of chunks on a PE.");
    app.add_flag(
        "--use-char-packing-samples",
        parameters.pdcx_config.use_char_packing_samples,
        "Pack multiple characters in the same datatype for phase 1 (samples) on the first level.");
    app.add_flag(
        "--use-char-packing-merging",
        parameters.pdcx_config.use_char_packing_merging,
        "Pack multiple characters in the same datatype for phase 4 (merging) on the first level.");
    app.add_flag("--rearrange-buckets-balanced",
                 parameters.pdcx_config.rearrange_buckets_balanced,
                 "Balances the buckets in a balanced way, which needs an additional output buffer "
                 "and some bookkeeping information.");
    app.add_flag("--use-robust-tie-break",
                 parameters.pdcx_config.use_robust_tie_break,
                 "Use ranks as a tie break in space efficient sorting in Phase 4. Is slower but "
                 "splits equal strings amoung buckets.");
    app.add_flag("--use-compressed-buckets",
                 parameters.pdcx_config.use_compressed_buckets,
                 "Store the bucket mapping compressed in the same memory as the SA.");
    app.add_option("--pack-extra-words",
                   parameters.pdcx_config.pack_extra_words,
                   "Use specificed number of extra words when packing characters into words. "
                   "Currently supports 0 and 1.");


    // sorter configuration
    app.add_option("--atomic-sorter",
                   parameters.pdcx_config.atomic_sorter,
                   "Atomic sorter to be used. [sample_sort, rquick, ams, bitonic, rfis]")
        ->transform(CLI::CheckedTransformer(string_map<mpi::AtomicSorters>(), CLI::ignore_case));
    app.add_option("--ams-levels",
                   parameters.pdcx_config.ams_levels,
                   "Number of levels to be used in ams.");

    app.add_option("--splitter-sampling",
                   parameters.pdcx_config.sample_sort_config.splitter_sampling,
                   "Splitter sampling method in sample sort. [uniform, random]")
        ->transform(
            CLI::CheckedTransformer(string_map<dsss::mpi::SplitterSampling>(), CLI::ignore_case));
    app.add_option("--splitter-sorting",
                   parameters.pdcx_config.sample_sort_config.splitter_sorting,
                   "Splitter sorting method in sample sort [central, distributed]")
        ->transform(
            CLI::CheckedTransformer(string_map<dsss::mpi::SplitterSorting>(), CLI::ignore_case));
    app.add_option("--string-sorter",
                   parameters.pdcx_config.string_sorter,
                   "String sorter to be used. [multi_key_qsort, radix_sort_ci2, radix_sort_ci3]")
        ->transform(CLI::CheckedTransformer(string_map<dsss::SeqStringSorter>(), CLI::ignore_case));
    app.add_option("--memory_seq_string_sorter",
                   parameters.pdcx_config.memory_seq_string_sorter,
                   "Memory hint for sequential string sorter.");

    app.add_flag("--use-string-sort",
                 parameters.pdcx_config.use_string_sort,
                 "Use string sorting instead of atomic sorting.");
    app.add_flag("--use-string-sort-tie-breaking-phase1",
                 parameters.pdcx_config.use_string_sort_tie_breaking_phase1,
                 "Use string sorting with index-tie-breaking in Phase 1.");
    app.add_flag("--use-string-sort-tie-breaking-phase4",
                 parameters.pdcx_config.use_string_sort_tie_breaking_phase4,
                 "Use string sorting with rank-tie-breaking in Phase 4.");
    app.add_flag("--use-loser-tree",
                 parameters.pdcx_config.sample_sort_config.use_loser_tree,
                 "Use loser tree in merging step of sample sort.");
    app.add_flag("--use-rquick-for-splitters",
                 parameters.pdcx_config.sample_sort_config.use_rquick_for_splitters,
                 "Use Rquick to sort splitter.");
    app.add_flag("--use-binary-search-for-splitters",
                 parameters.pdcx_config.sample_sort_config.use_binary_search_for_splitters,
                 "Use binary search instead of linear scan to find intervals in sample sort.");
    app.add_flag("--use-lcp-compression",
                 parameters.pdcx_config.sample_sort_config.use_lcp_compression,
                 "Use lcp-compression in string sample sort to reduce communication volume.");
    app.add_option("--lcp-compression-threshold",
                   parameters.pdcx_config.sample_sort_config.lcp_compression_threshold,
                   "Value between [0, 1], threshold on compression ratio when to start using "
                   "LCP-compression.");
    app.add_flag("--use-prefix-doubling",
                 parameters.pdcx_config.sample_sort_config.use_prefix_doubling,
                 "Use prefix-doubling in string sample sort to reduce communication volume.");
    app.add_option("--inital-prefix-length",
                   parameters.pdcx_config.sample_sort_config.inital_prefix_length,
                   "Inital prefix-length to use for prefix doubling.");

    CLI11_PARSE_MPI(app, argc, argv);
    parameters.num_pe = kamping::comm_world().size();
    return parameters;
}

template <typename EnumType>
EnumType get_enum(std::string s, std::vector<std::string> names, kamping::Communicator<>& comm) {
    for (uint i = 0; i < names.size(); i++) {
        if (s == names[i]) {
            return static_cast<EnumType>(i);
        }
    }
    if (comm.rank() == 0) {
        std::cout << "Invalid enum: " << s << std::endl;
        std::cout << "Available options: ";
        bool is_first = true;
        for (std::string& s: names) {
            if (!is_first) {
                std::cout << ", ";
            }
            std::cout << s;
            is_first = false;
        }
        std::cout << std::endl;
    }
    exit(1);
}

void map_strings_to_enum(kamping::Communicator<>& comm) {
    //// pdcx
    // pdcx_config.atomic_sorter = magic_enum::enum_cast<mpi::AtomicSorters>(atomic_sorter).value();
    //// get_enum<mpi::AtomicSorters>(atomic_sorter, mpi::atomic_sorter_names, comm);
    // pdcx_config.string_sorter =
    // magic_enum::enum_cast<dsss::SeqStringSorter>(string_sorter).value();
    //// get_enum<dsss::SeqStringSorter>(string_sorter, dsss::string_sorter_names, comm);

    //// sample sort
    // sample_sort_config.splitter_sorting =
    //     magic_enum::enum_cast<dsss::mpi::SplitterSorting>(splitter_sorting).value();
    //// get_enum<dsss::mpi::SplitterSorting>(splitter_sorting,
    ////                                      dsss::mpi::splitter_sorting_names,
    ////                                      comm);
    // sample_sort_config.splitter_sampling =
    //     magic_enum::enum_cast<dsss::mpi::SplitterSampling>(splitter_sampling).value();
    //// get_enum<dsss::mpi::SplitterSampling>(splitter_sampling,
    ////                                       dsss::mpi::splitter_sampling_names,
    ////                                       comm);

    // pdcx_config.sample_sort_config = sample_sort_config;
}

std::vector<uint32_t> parse_list_of_ints(std::string s) {
    char separator = ',';
    std::replace(s.begin(), s.end(), separator, ' ');

    std::vector<uint32_t> numbers;
    std::stringstream ss(s);
    uint32_t temp;
    while (ss >> temp) {
        numbers.push_back(temp);
    }
    return numbers;
}


void parse_enums_and_lists(kamping::Communicator<>& comm) {
    // map_strings_to_enum(comm);
    // pdcx_config.buckets_samples = parse_list_of_ints(buckets_samples);
    // pdcx_config.buckets_merging = parse_list_of_ints(buckets_merging);
    // uint64_t limit = std::numeric_limits<uint16_t>::max();
    // check_limit(pdcx_config.buckets_samples, limit, "buckets_samples", comm);
    // check_limit(pdcx_config.buckets_merging, limit, "buckets_merging", comm);
}

void report_arguments(kamping::Communicator<>& comm) {
    comm.barrier();
    if (comm.rank() == 0) {
        std::cout << "Arguments:\n";
        std::cout << V(string_size) << "\n";
        std::cout << V(alphabet_size) << "\n";
        std::cout << V(seed) << "\n";
        std::cout << V(input_path) << "\n";
        std::cout << V(output_path) << "\n";
        std::cout << V(dcx_variant) << "\n";
        std::cout << V(check) << "\n";
        std::cout << std::endl;
        // pdcx_config.print_config();
    }
    comm.barrier();
}

void read_input(kamping::Communicator<>& comm, Parameters const& parameters) {
    if (parameters.input_path != "random" && !mpi::file_exists(parameters.input_path)) {
        if (comm.rank() == 0) {
            std::cerr << "File " << parameters.input_path << " does not exist!" << std::endl;
        }
        exit(1);
    }
    auto& timer = kamping::measurements::timer();
    timer.synchronize_and_start("io");
    if (!input_path.compare("random")) {
        string_size /= comm.size();
        uint64_t local_seed = seed + comm.rank();
        local_string = random::generate_random_data<char_type>(parameters.textsize,
                                                               parameters.alphabet_size,
                                                               local_seed);
    } else {
        local_string = mpi::read_and_distribute_string<char_type>(parameters.input_path,
                                                                  comm,
                                                                  parameters.textsize);
    }
    timer.stop();
    kamping::report_on_root("\n", comm);
}

void compress_alphabet(std::vector<char_type>& input, kamping::Communicator<>& comm) {
    uint64_t max_alphabet_size = 256;

    // should not happen, because we read characters as bytes
    uint64_t max_char = mpi_util::all_reduce_max(input, comm);
    if (max_char > max_alphabet_size) {
        kamping::report_on_root(
            "Can only process alphabets with not more than 255 distinct "
            "characters. 0 is reserved for special characters. Change char_type.",
            comm);
        exit(1);
    }

    // determine character frequencies
    std::vector<uint64_t> local_counts(max_alphabet_size, 0);
    for (auto c: input) {
        local_counts[c]++;
    }
    std::vector<uint64_t> global_counts =
        comm.allreduce(kamping::send_buf(local_counts), kamping::op(kamping::ops::plus<>{}));
    uint64_t alphabet_size =
        local_counts.size() - std::count(global_counts.begin(), global_counts.end(), 0);

    if (alphabet_size == local_counts.size()) {
        kamping::report_on_root(
            "Can only process alphabets with not more than 255 distinct "
            "characters. 0 is reserved for special characters. Change char_type.",
            comm);
        exit(1);
    }

    // reserve character 0 for padding
    uint64_t next_char = 1;
    std::vector<uint64_t> map_char(max_alphabet_size);
    for (uint64_t i = 0; i < max_alphabet_size; i++) {
        if (global_counts[i] > 0) {
            map_char[i] = next_char++;
        }
    }

    // map input alphabet to compressed alphabet
    for (uint64_t i = 0; i < input.size(); i++) {
        input[i] = map_char[input[i]];
    }
    kamping::report_on_root("input_alphabet_size=" + std::to_string(alphabet_size), comm);
    input_alphabet_size = alphabet_size;
}

template <typename PDCX, typename char_type, typename index_type>
void run_pdcx(kamping::Communicator<>& comm, const dsss::dcx::PDCXConfig& pdcx_config) {
    auto algo = PDCX(pdcx_config, comm);
    local_sa = algo.compute_sa(local_string);
    algo.report_time();
    kamping::report_on_root("\n", comm);
    algo.report_stats();
}


template <typename DCXParam, uint64_t EXTRA_WORDS = 0>
void run_packed_dcx_variant(kamping::Communicator<>& comm,
                            dsss::dcx::PDCXConfig const& pdcx_config) {
    using namespace dcx;
    using WordType = uint64_t;
    constexpr uint64_t X = DCXParam::X;
    constexpr uint64_t BITS_WORD = 8 * sizeof(WordType);

    // logging
    uint64_t packed_chars;
    uint64_t bits_per_char;
    double packing_ratio;

    if (input_alphabet_size <= (1 << 3) - 1) {
        // 3-bit variant
        constexpr uint64_t BITS_CHAR = 3;
        constexpr uint64_t CHARS_PER_WORD = BITS_WORD / BITS_CHAR;
        constexpr uint64_t NUM_WORDS = ((X + CHARS_PER_WORD - 1) / CHARS_PER_WORD) + EXTRA_WORDS;
        // constexpr uint64_t NUM_WORDS = 2; // TEMPORARY for dc21 and dc31
        constexpr uint64_t PACKED_CHARS = NUM_WORDS * CHARS_PER_WORD;
        using CharContainer = KPackedInteger<NUM_WORDS, char_type, BITS_CHAR, WordType>;
        using PDCXVariant = PDCX<char_type, index_type, DCXParam, CharContainer, CharContainer>;


        pdcx_config.packing_ratio = (double)PACKED_CHARS / X;
        packed_chars = PACKED_CHARS;
        bits_per_char = BITS_CHAR;
        packing_ratio = pdcx_config.packing_ratio;

        run_pdcx<PDCXVariant, char_type, index_type>(comm, pdcx_config);
    } else if (input_alphabet_size <= (1 << 5) - 1) {
        // 5-bit variant
        constexpr uint64_t BITS_CHAR = 5;
        constexpr uint64_t CHARS_PER_WORD = BITS_WORD / BITS_CHAR;
        constexpr uint64_t NUM_WORDS = ((X + CHARS_PER_WORD - 1) / CHARS_PER_WORD) + EXTRA_WORDS;
        constexpr uint64_t PACKED_CHARS = NUM_WORDS * CHARS_PER_WORD;
        using CharContainer = KPackedInteger<NUM_WORDS, char_type, BITS_CHAR, WordType>;
        using PDCXVariant = PDCX<char_type, index_type, DCXParam, CharContainer, CharContainer>;

        pdcx_config.packing_ratio = (double)PACKED_CHARS / X;
        packed_chars = PACKED_CHARS;
        bits_per_char = BITS_CHAR;
        packing_ratio = pdcx_config.packing_ratio;
        run_pdcx<PDCXVariant, char_type, index_type>(comm, pdcx_config);
    } else {
        // 8-bit variant
        constexpr uint64_t BITS_CHAR = 8;
        constexpr uint64_t CHARS_PER_WORD = BITS_WORD / BITS_CHAR;
        constexpr uint64_t NUM_WORDS = ((X + CHARS_PER_WORD - 1) / CHARS_PER_WORD) + EXTRA_WORDS;
        constexpr uint64_t PACKED_CHARS = NUM_WORDS * CHARS_PER_WORD;
        using CharContainer = KPackedInteger<NUM_WORDS, char_type, BITS_CHAR, WordType>;
        using PDCXVariant = PDCX<char_type, index_type, DCXParam, CharContainer, CharContainer>;

        pdcx_config.packing_ratio = (double)PACKED_CHARS / X;
        packed_chars = PACKED_CHARS;
        bits_per_char = BITS_CHAR;
        packing_ratio = pdcx_config.packing_ratio;
        run_pdcx<PDCXVariant, char_type, index_type>(comm, pdcx_config);
    }

    // logging
    report_on_root("packed_chars=" + std::to_string(packed_chars), comm);
    report_on_root("_packing_ratio=" + std::to_string(packing_ratio), comm);
    report_on_root("bits_per_char=" + std::to_string(bits_per_char), comm);
}

void select_dcx_variant(kamping::Communicator<>& comm, dsss::dcx::PDCXConfig const& pdcx_config) {
    using namespace dcx;

#ifdef DCX_BUILD_ALL
    if (dcx_variant == "dc3") {
        using DCXParam = DC3Param;
        run_pdcx<PDCX<char_type, index_type, DCXParam>, char_type, index_type>(comm, pdcx_config);
        return;
    } else if (dcx_variant == "dc7") {
        using DCXParam = DC7Param;
        return;
        run_pdcx<PDCX<char_type, index_type, DCXParam>, char_type, index_type>(comm, pdcx_config);
    } else if (dcx_variant == "dc13") {
        using DCXParam = DC13Param;
        run_pdcx<PDCX<char_type, index_type, DCXParam>, char_type, index_type>(comm, pdcx_config);
        return;
    } else if (dcx_variant == "dc21") {
        using DCXParam = DC21Param;
        run_pdcx<PDCX<char_type, index_type, DCXParam>, char_type, index_type>(comm, pdcx_config);
        return;
    } else if (dcx_variant == "dc31") {
        using DCXParam = DC31Param;
        run_pdcx<PDCX<char_type, index_type, DCXParam>, char_type, index_type>(comm, pdcx_config);
        return;
    } else if (dcx_variant == "dc39") {
        using DCXParam = DC39Param;
        run_pdcx<PDCX<char_type, index_type, DCXParam>, char_type, index_type>(comm, pdcx_config);
        return;
    } else if (dcx_variant == "dc57") {
        using DCXParam = DC57Param;
        run_pdcx<PDCX<char_type, index_type, DCXParam>, char_type, index_type>(comm, pdcx_config);
        return;
    } else if (dcx_variant == "dc73") {
        using DCXParam = DC73Param;
        run_pdcx<PDCX<char_type, index_type, DCXParam>, char_type, index_type>(comm, pdcx_config);
        return;
    } else if (dcx_variant == "dc91") {
        using DCXParam = DC91Param;
        run_pdcx<PDCX<char_type, index_type, DCXParam>, char_type, index_type>(comm, pdcx_config);
        return;
    } else if (dcx_variant == "dc95") {
        using DCXParam = DC95Param;
        run_pdcx<PDCX<char_type, index_type, DCXParam>, char_type, index_type>(comm, pdcx_config);
        return;
    } else if (dcx_variant == "dc133") {
        using DCXParam = DC133Param;
        run_pdcx<PDCX<char_type, index_type, DCXParam>, char_type, index_type>(comm, pdcx_config);
        return;
    } else {
        using DCXParam = DC39Param;
        using PDCXVariant = PDCX<char_type, index_type, DCXParam>;
        run_pdcx<PDCXVariant, char_type, index_type>(comm, pdcx_config);
        return;
    }
#endif
}

template <uint64_t EXTRA_WORDS = 0>
void select_packed_dcx_variant(kamping::Communicator<>& comm,
                               dsss::dcx::PDCXConfig const& pdcx_config) {
    using namespace dcx;

#ifdef DCX_BUILD_ALL
    if (dcx_variant == "dc3") {
        using DCXParam = DC3Param;
        run_packed_dcx_variant<DCXParam, EXTRA_WORDS>(comm, pdcx_config);
        return;
    } else if (dcx_variant == "dc7") {
        using DCXParam = DC7Param;
        run_packed_dcx_variant<DCXParam, EXTRA_WORDS>(comm, pdcx_config);
        return;
    } else if (dcx_variant == "dc13") {
        using DCXParam = DC13Param;
        run_packed_dcx_variant<DCXParam, EXTRA_WORDS>(comm, pdcx_config);
        return;
    } else if (dcx_variant == "dc21") {
        using DCXParam = DC21Param;
        run_packed_dcx_variant<DCXParam, EXTRA_WORDS>(comm, pdcx_config);
        return;
    } else if (dcx_variant == "dc31") {
        using DCXParam = DC31Param;
        run_packed_dcx_variant<DCXParam, EXTRA_WORDS>(comm, pdcx_config);
        return;
    } else if (dcx_variant == "dc39") {
        using DCXParam = DC39Param;
        run_packed_dcx_variant<DCXParam, EXTRA_WORDS>(comm, pdcx_config);
        return;
    } else if (dcx_variant == "dc57") {
        using DCXParam = DC57Param;
        run_packed_dcx_variant<DCXParam, EXTRA_WORDS>(comm, pdcx_config);
        return;
    }
    else if (dcx_variant == "dc73") {
        using DCXParam = DC73Param;
        run_packed_dcx_variant<DCXParam, EXTRA_WORDS>(comm, pdcx_config);
        return;
    } else if (dcx_variant == "dc91") {
        using DCXParam = DC91Param;
        run_packed_dcx_variant<DCXParam, EXTRA_WORDS>(comm, pdcx_config);
        return;
    } else if (dcx_variant == "dc95") {
        using DCXParam = DC95Param;
        run_packed_dcx_variant<DCXParam, EXTRA_WORDS>(comm, pdcx_config);
        return;
    } else if (dcx_variant == "dc133") {
        using DCXParam = DC133Param;
        run_packed_dcx_variant<DCXParam, EXTRA_WORDS>(comm, pdcx_config);
        return;
    }
#endif
    using DCXParam = DC39Param;
    run_packed_dcx_variant<DCXParam, 0>(comm, pdcx_config);
}

void compute_sa(kamping::Communicator<>& comm, dsss::dcx::PDCXConfig const& pdcx_config) {
    using namespace dcx;

    measurements::Timer<Communicator<>> algo_timer;
    auto& timer = kamping::measurements::timer();
    timer.synchronize_and_start("total_time");
    algo_timer.synchronize_and_start("total_time");


    timer.synchronize_and_start("compress_alphabet");
    compress_alphabet(local_string, comm);
    timer.stop();


    if (pdcx_config.use_char_packing_merging || pdcx_config.use_char_packing_samples) {
        /*** better variant with packed integers  ***/
        if (pdcx_config.pack_extra_words == 0) {
            constexpr uint64_t EXTRA_WORDS = 0;
            select_packed_dcx_variant<EXTRA_WORDS>(comm, pdcx_config);
        }
        //  else {
        //     constexpr uint64_t EXTRA_WORDS = 1;
        //     select_packed_dcx_variant<EXTRA_WORDS>(comm);
        // }

    } else {
        /*** standard variant with atomic sorting or string sorting  ***/
        select_dcx_variant(comm, pdcx_config);
    }

    algo_timer.stop();
    timer.stop();
    algo_timer.aggregate_and_print(kamping::measurements::FlatPrinter{});
    kamping::report_on_root("\n", comm);
}

void write_sa(kamping::Communicator<>& comm, Parameters const& params) {
    if (!params.output_path.empty()) {
        kamping::report_on_root("Writing the SA to " + params.output_path + "\n", comm);
        mpi::write_data(local_sa, params.output_path, comm);
        comm.barrier();
        kamping::report_on_root("Finished writing the SA \n", comm);
    }
}

void check_sa(kamping::Communicator<>& comm, Parameters const& params) {
    using namespace kamping;
    if (params.check) {
        measurements::Timer<Communicator<>> check_timer;
        measurements::timer().synchronize_and_start("check_SA");
        check_timer.synchronize_and_start("check_SA");

        // TODO maybe read again
        read_input(comm, params);

        kamping::report_on_root("Checking SA ... ", comm);
        // assuming algorithm did not change local string
        bool correct = check_suffixarray(local_sa, local_string, comm);
        // bool correct = check_suffixarray2(local_sa, local_string, comm);
        // bool correct2 = check_suffixarray2(local_sa, local_string, comm);

        if (comm.rank() == 0) {
            std::string msg = correct ? "Correct SA!" : "ERROR: Not a correct SA!";
            std::cout << msg << std::endl;
            std::cout << "SA_ok=" << correct << std::endl;
            // std::cout << "SA_ok2=" << correct2 << std::endl;
        }
        check_timer.stop();
        measurements::timer().stop();
        check_timer.aggregate_and_print(kamping::measurements::FlatPrinter{});
        kamping::report_on_root("\n", comm);
    }
}

void report_memory_usage(kamping::Communicator<>& comm, bool output_rss_from_all_pes = false) {
    kamping::report_on_root("Memory Usage:", comm);
    uint64_t max_mem = dsss::get_max_mem_bytes();
    uint64_t max_rss = mpi_util::all_reduce_max(max_mem, comm);
    double blowup = (double)max_rss / local_string.size();
    kamping::report_on_root("max_rss_pe=" + std::to_string(max_rss), comm);
    kamping::report_on_root("blowup_pe=" + std::to_string(blowup), comm);

    if (output_rss_from_all_pes) {
        auto all_mem = comm.gather(kamping::send_buf(max_mem));
        if (comm.rank() == 0) {
            std::cout << "max_mem_pe=";
            kamping::print_vector(all_mem, ",");
            std::cout << std::endl;
        }
    }
}

int main(int32_t argc, char const* argv[]) {
    uint64_t max_mem_start = dsss::get_max_mem_bytes();
    kamping::Environment e;
    kamping::Communicator comm;
    kamping::measurements::counter().clear();
    uint64_t max_mem_init = dsss::get_max_mem_bytes();
    kamping::measurements::counter().add("mem_program_start",
                                         max_mem_start,
                                         {kamping::measurements::GlobalAggregationMode::max,
                                          kamping::measurements::GlobalAggregationMode::gather});

    kamping::measurements::counter().add("mem_program_after_init",
                                         max_mem_init,
                                         {kamping::measurements::GlobalAggregationMode::max,
                                          kamping::measurements::GlobalAggregationMode::gather});
    options::report_compile_flags(comm);

    Parameters const params = read_cli_parameters(argc, argv);
    dcx_variant = params.dcx_variant;
    // if (!cp.process(argc, argv)) {
    //     return -1;
    // }
    // parse_enums_and_lists(comm);
    // report_arguments(comm);


    uint64_t max_mem_before_input = dsss::get_max_mem_bytes();
    auto& timer = kamping::measurements::timer();
    timer.clear();
    kamping::measurements::counter().add("mem_before_reading_input",
                                         max_mem_before_input,
                                         {kamping::measurements::GlobalAggregationMode::max,
                                          kamping::measurements::GlobalAggregationMode::gather});
    read_input(comm, params);
    kamping::measurements::counter().add("mem_before_sa_construction",
                                         dsss::get_max_mem_bytes(),
                                         {kamping::measurements::GlobalAggregationMode::max,
                                          kamping::measurements::GlobalAggregationMode::gather});

    compute_sa(comm, params.pdcx_config);
    dcx::get_local_stats_instance().commit();
    dcx::get_local_stats_instance().reset();
    kamping::measurements::counter().add("mem_after_sa_construction",
                                         dsss::get_max_mem_bytes(),
                                         {kamping::measurements::GlobalAggregationMode::max,
                                          kamping::measurements::GlobalAggregationMode::gather});
    report_memory_usage(comm);
    check_sa(comm, params);

    kamping::measurements::counter().add("mem_after_sa_check",
                                         dsss::get_max_mem_bytes(),
                                         {kamping::measurements::GlobalAggregationMode::max,
                                          kamping::measurements::GlobalAggregationMode::gather});

    write_sa(comm, params);

    // print
    auto config_vector = params.config();
    std::stringstream sstream_counter;
    std::stringstream sstream_timer;
    kamping::measurements::SimpleJsonPrinter<double> printer_timer(sstream_timer, config_vector);
    kamping::measurements::SimpleJsonPrinter<std::int64_t> printer_counter(sstream_counter,
                                                                           config_vector);
    kamping::measurements::timer().aggregate_and_print(printer_timer);
    kamping::measurements::counter().aggregate_and_print(printer_counter);

    if (comm.rank() == 0) {
        print_as_jsonlist_to_file({sstream_timer.str()}, params.json_output_path + "_timer.json");
        print_as_jsonlist_to_file({sstream_counter.str()},
                                  params.json_output_path + "_counter.json");
    }

    return 0;
}
