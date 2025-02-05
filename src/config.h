#ifndef VANITY_CONFIG
#define VANITY_CONFIG

#include <cuda_runtime.h>
#include <stdio.h>

static int MAX_ITERATIONS = 100000;
static int STOP_AFTER_KEYS_FOUND = 100;

// how many times a gpu thread generates a public key in one go
static __device__ int ATTEMPTS_PER_EXECUTION;
static int attempts_per_exec = 100000;

static __device__ const int MAX_PATTERNS = 10;

// Arrays for both prefix and suffix patterns
static __device__ char prefixes[MAX_PATTERNS][32];
static __device__ char suffixes[MAX_PATTERNS][32];

// Counters for number of patterns
static __device__ int prefix_count_device;
static __device__ int suffix_count_device;

static __device__ __constant__ FILE *output_file;

#endif
