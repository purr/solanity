#ifndef VANITY_CONFIG
#define VANITY_CONFIG

static int MAX_ITERATIONS = 100000;
static int STOP_AFTER_KEYS_FOUND = 100;

// how many times a gpu thread generates a public key in one go
__device__ int ATTEMPTS_PER_EXECUTION;
static int attempts_per_exec = 100000;

__device__ const int MAX_PATTERNS = 10;

__device__ bool is_suffix_match = false;

// Prefix array that will be populated from command line args
__device__ char prefixes[MAX_PATTERNS][32];

__device__ __constant__ FILE* output_file;

#endif
