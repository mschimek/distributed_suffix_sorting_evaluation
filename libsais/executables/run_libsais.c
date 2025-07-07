// This file has been added by Manuel Haag, Matthias Schimek, 2025

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/resource.h>

// 64-bit version
#include "../include/libsais64.h"

#include <time.h>

#include <getopt.h>
#include <string.h>
#include <errno.h>

typedef struct {
    int     iteration;
    char   *input_path;
    char   *json_output_path;
    long long    textsize;
    int     num_threads;
    int     num_pe;
} Config;

double get_wall_time() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

char *make_txt_filename(const char *base, const char* suffix) {
    size_t len  = strlen(base);
    size_t slen = strlen(suffix);
    char *out = malloc(len + slen + 1);
    if (!out) {
        perror("malloc failed");
        return NULL;
    }
    strcpy(out, base);
    strcat(out, suffix);
    return out;
}

void write_output(const char *filename,
                  long num_pe,
                  long num_threads,
                  long long textsize,
                  const char *input_path,
                  const char *algorithm,
                  long external_iteration,
                  double total_time)
{
    char* new_filename = make_txt_filename(filename, "_timer.json");
    FILE *fp = fopen(new_filename, "w");
    if (!fp) {
        fprintf(stderr, "Error opening '%s': %s\n", filename, strerror(errno));
        return;
    }

    fprintf(fp,
        "[\n"
        "{\n"
        "  \"data\": {\n"
        "    \"root\": {\n"
        "      \"statistics\": {\n"
        "      },\n"
        "      \"total_time\": {\n"
        "        \"statistics\": {\n"
        "          \"max\": [%f]\n"
        "        }\n"
        "      }\n"
        "    }\n"
        "  },\n"
        "  \"config\": {\n"
        "    \"num_pe\":\"%ld\",\n"
        "    \"num_threads\":\"%ld\",\n"
        "    \"textsize\":\"%lld\",\n"
        "    \"input_path\":\"%s\",\n"
        "    \"algorithm\":\"%s\",\n"
        "\n"
        "    \"external_iteration\":\"%ld\"\n"
        "  }\n"
        "}\n"
        "]\n",
        total_time,
        num_pe,
        num_threads,
        textsize,
        input_path,
        algorithm,
        external_iteration
    );

    fclose(fp);
    free(new_filename);
}
void write_counter_output(const char *filename,
                  long num_pe,
                  long num_threads,
                  long long textsize,
                  const char *input_path,
                  const char *algorithm,
                  long external_iteration,
                  long long memory_consumption)
{
    char* new_filename = make_txt_filename(filename, "_counter.json");
    FILE *fp = fopen(new_filename, "w");
    if (!fp) {
        fprintf(stderr, "Error opening '%s': %s\n", filename, strerror(errno));
        return;
    }

    fprintf(fp,
        "[\n"
        "{\n"
        "  \"data\": {\n"
        "    \"root\": {\n"
        "      \"statistics\": {\n"
        "      },\n"
        "      \"memory_consumption_after_sa_construction\": {\n"
        "        \"statistics\": {\n"
        "          \"max\": [%lld]\n"
        "        }\n"
        "      }\n"
        "    }\n"
        "  },\n"
        "  \"config\": {\n"
        "    \"num_pe\":\"%ld\",\n"
        "    \"num_threads\":\"%ld\",\n"
        "    \"textsize\":\"%lld\",\n"
        "    \"input_path\":\"%s\",\n"
        "    \"algorithm\":\"%s\",\n"
        "\n"
        "    \"external_iteration\":\"%ld\"\n"
        "  }\n"
        "}\n"
        "]\n",
        memory_consumption,
        num_pe,       
        num_threads,   
        textsize,       
        input_path,      
        algorithm,        
        external_iteration 
    );

    fclose(fp);
    free(new_filename);
}

void print_usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s  --iteration N  --input PATH  --json_output_path PATH  \\\n"
        "           --textsize S  --num_threads T\n", prog);
    exit(EXIT_FAILURE);
}

void parse_args(int argc, char *argv[], Config *cfg) {
    static struct option long_options[] = {
        {"iteration",         required_argument, 0, 'i'},
        {"input",             required_argument, 0, 'n'},
        {"json_output_path",  required_argument, 0, 'j'},
        {"textsize",          required_argument, 0, 't'},
        {"num_threads",       required_argument, 0, 'p'},
        {"num_pe",            required_argument, 0, 'q'},
        {0, 0, 0, 0}
    };

    int opt, option_index = 0;
    cfg->iteration        = -1;
    cfg->input_path       = NULL;
    cfg->json_output_path = NULL;
    cfg->textsize         = -1;
    cfg->num_threads      = -1;
    cfg->num_pe           = -1;

    while ((opt = getopt_long(argc, argv, "", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'i':
                cfg->iteration = atoi(optarg);
                break;
            case 'n':
                cfg->input_path = strdup(optarg);
                break;
            case 'j':
                cfg->json_output_path = strdup(optarg);
                break;
            case 't':
                cfg->textsize = atoll(optarg);
                break;
            case 'p':
                cfg->num_threads = atoi(optarg);
                break;
            case 'q':
                cfg->num_pe = atoi(optarg);
                break;
            default:
                print_usage(argv[0]);
        }
    }

    if (cfg->iteration < 0 || !cfg->input_path ||
        !cfg->json_output_path || cfg->textsize < 0 ||
        cfg->num_threads < 0) {
        print_usage(argv[0]);
    }
}

int main(int argc, char *argv[]) {
    Config cfg;
    parse_args(argc, argv, &cfg);

    FILE *file_ptr = fopen(cfg.input_path, "rb");
    fseek(file_ptr, 0, SEEK_END);
    size_t file_size = ftell(file_ptr);
    fseek(file_ptr, 0, SEEK_SET);
    if (!file_ptr) {
        perror("Error opening file");
        return EXIT_FAILURE;
    }

    long long n_bytes = cfg.textsize;
    if(n_bytes > file_size) {
        n_bytes = file_size;
    }
    #ifdef LIBSAIS_USE_OPENMP
    int use_omp = 1;
    #else
    int use_omp = 0;
    #endif
    printf("input_file=%s\n", argv[1]);
    printf("input_size_bytes=%ld\n", n_bytes);
    printf("use_omp=%d\n", use_omp);
    
    // Allocate buffer for reading bytes
    char *text = malloc(n_bytes);
    if (!text) {
        fprintf(stderr, "Memory allocation failed\n");
        fclose(file_ptr);
        return EXIT_FAILURE;
    }




    long threads = cfg.num_threads;
    size_t bytes_read = fread(text, 1, n_bytes, file_ptr);
    
    int64_t *SA = malloc(sizeof(int64_t) * n_bytes);

    clock_t start = clock();
    
    // LIBSAIS
    printf("Starting Libsais...\n");
    double start1 = get_wall_time();
    if(threads == 1) {
    	libsais64(text, SA, n_bytes, 0, NULL);
    } else {
    	libsais64_omp(text, SA, n_bytes, 0, NULL, threads);
    }
    double end1 = get_wall_time();
    printf("Libsais finished.\n");
    // LIBSAIS
    
    clock_t end = clock();
    double elapsed_time_s = (double)(end - start) / CLOCKS_PER_SEC;
    double elapsed_time_s1 = ((double)(end1 - start1));
    
    printf("Execution time: %.9f seconds\n", elapsed_time_s);

    printf("\tExecution time1: %.9f seconds\n", elapsed_time_s1);
    printf("total_time=%.9f\n", elapsed_time_s);

    struct rusage r_usage;
    getrusage(RUSAGE_SELF,&r_usage);
    printf("Memory usage: %ld kilobytes\n",r_usage.ru_maxrss);
    printf("max_mem_bytes=%ld\n",r_usage.ru_maxrss * 1000);
    
    double n_kbytes = (double)n_bytes / 1000.;
    double n_mbytes = (double)n_bytes / 1000000.;
    double throughput = n_mbytes / elapsed_time_s1; // MB / s
    double blowup = r_usage.ru_maxrss / n_kbytes;
    printf("Throughput: %f MB/s\n", throughput);
    printf("throughput=%f\n", throughput);
    printf("blowup=%f\n", blowup);

    write_output(cfg.json_output_path, cfg.num_pe, cfg.num_threads, cfg.textsize, cfg.input_path, "LIBSAIS", cfg.iteration, elapsed_time_s1);
    write_counter_output(cfg.json_output_path, cfg.num_pe, cfg.num_threads, cfg.textsize, cfg.input_path, "LIBSAIS", cfg.iteration, (r_usage.ru_maxrss * ((long long)1000)));

    // Clean up
    free(text);
    free(SA);
    fclose(file_ptr);
    free(cfg.input_path);
    free(cfg.json_output_path);

    return EXIT_SUCCESS;
}
