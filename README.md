# Solana Vanity Address Generator

A high-performance CUDA-based vanity address generator for Solana. Generates ed25519 keypairs matching specified patterns.

## Quick Start

```bash
# First time setup
chmod +x mk      # Make build script executable
chmod +x vanity  # Make wrapper script executable
chmod +x detect_gpu.py  # Make GPU detection script executable

# If CUDA is not in your system PATH, edit mk script and uncomment:
# export PATH=/usr/local/cuda/bin:$PATH

# Build the project
./mk   # This will automatically detect your GPU architecture before building

# Run with simple pattern
./vanity -p ABC        # Find address starting with ABC
./vanity -s XYZ        # Find address ending with XYZ

# Multiple patterns (matches any of them)
./vanity -p ABC -p DEF  # Find address starting with ABC or DEF
./vanity -s XYZ -s 123  # Find address ending with XYZ or 123

# Combine prefix and suffix
./vanity -p ABC -s XYZ  # Find address starting with ABC AND ending with XYZ

# Use wildcards
./vanity -p A?C        # ? matches any character
```

## Options

- `-p, --prefix PATTERN` - Add a prefix pattern to match (can be used multiple times)
- `-s, --suffix PATTERN` - Add a suffix pattern to match (can be used multiple times)
- `-n, --num-keys NUM` - Number of keys to generate before stopping (default: 100)
- `-i, --iterations NUM` - Maximum number of GPU kernel launches (default: 1000000). Each iteration represents one batch of parallel work on the GPU.
- `-a, --attempts NUM` - Number of key generation attempts per GPU thread in each kernel execution (default: 1000000). Higher values mean more work per GPU launch but may affect responsiveness.

For example, with default settings on a GPU with 1000 threads:

- Each thread tries 1000000 keys per kernel launch
- The kernel launches up to 1000000 times
- Total possible attempts = threads × attempts × iterations
- Program stops early if the desired number of keys is found

## Performance

Using a single Tesla V100:

- Simple pattern (e.g., "AAAA"): < 10 seconds
- Five-letter pattern (e.g., "AAAAA"): ~12 minutes
- Multiple patterns: Almost no performance penalty

## Important Notes

1. Pattern matching is case-sensitive ("ABC" will not match "abc")
2. Use ? as a wildcard to match any single character
3. While this tool uses CUDA's random number generator (CURAND) for key generation, it is not recommended for production use as CURAND's XORWOW algorithm is not cryptographically secure
4. The program will keep running until it finds the specified number of matches or reaches the maximum iterations

## Output Format

Matches are printed in the format:

```
MATCH:PublicKey,PrivateKeySeed
```

Example:

```
MATCH:ABCDEFGHIJKLMNOPQRSTUVWXYZabcd,1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef
```

## Converting to Solana Keypair

The program includes a Python script to convert the hex output to a Solana keypair file:

```bash
# Install required packages
pip install base58 pynacl

# Convert using command line
python convert.py <hex_string>

# Or run without arguments for interactive input
python convert.py
```

The script will:

1. Convert the hex seed to a proper Solana keypair
2. Save it in a file named `keypair_PUBLICKEY.json`, where PUBLICKEY is the base58-encoded public key
3. Display the public key for verification

You can then use this keypair file with Solana CLI tools.

# Find a Solana vanity address using GPUs

I originally copied this from here: https://github.com/ChorusOne/solanity

Then I made the following changes:

1. Initialise the search using entropy from the OS (it was deterministic)
2. Exact matches only (there was some weird lowercase thing going on)
3. Exit criteria based on number of keys found or iterations
4. Output Solana keypair in log

When it finds a match, it will log a line starting with MATCH, you will see the vanity address found and the secret (seed) in hex.

A Solana keypair file is a text file with one line, that has 64 bytes as decimal numbers in json format. The first 32 bytes are the (secret) seed, the last 32 bytes are the public key. This public key, when represented in base58 format, is the (vanity) address. The line you are looking for is immediatley after the match line, something like this:

[59,140,24,207,208,39,85,22,191,118,230,168,183,34,21,196,25,202,215,167,74,68,74,29,50,247,170,102,19,66,27,104,136,17,198,97,155,247,112,195,114,159,140,43,11,156,171,32,112,188,1,46,231,106,16,148,200,105,30,83,19,235,139,5]

Paste this one line into a file keypair.json for example, and test it by sending funds to and from it.

Using a single Tesla V100 searching for AAAA only, you should find a match under 10 seconds, for AAAAA average time is 12 minutes.

There is almost zero penalty for searching multiple prefix patterns, so you should.

nb: I changed the logging of 'attempts', the original author thought it should be multiplied by 64, but I have no idea why, so I removed that.

NO WARRANTY OR LIABILITY WHATSOEVER, IN THIS WORLD OR THE WORLD TO COME, YOUR SOUL IS YOUR OWN RESPONSIBILITY.

The original instructions are reproduced below and are correct for build:

# A CUDA based ed25519 vanity key finder (in Base58)

This is a GPU based vanity key finder. It does not currently use a CSPRNG and
any key generated by this tool is 100% not secure to use. Great fun for Tour de
Sol though.

## Configure

Open `src/config.h` and add any prefixes you want to scan for to the list.

## Building

The project includes automatic GPU architecture detection:

1. The `detect_gpu.py` script uses nvidia-smi to identify your GPU
2. It automatically updates the build configuration in `src/gpu-common.mk`
3. The `mk` script runs this detection before each build
4. No manual configuration of GPU architecture is needed

Requirements:

- CUDA toolkit installed
- nvidia-smi available in PATH
- Python 3.x installed

If the automatic detection fails, you can manually edit `src/gpu-common.mk` with your GPU architecture.

Make sure your cuda binary are in your path, and build:

```bash
$ export PATH=/usr/local/cuda/bin:$PATH
$ make -j$(nproc)
```

## Running

```bash
LD_LIBRARY_PATH=./src/release ./src/release/cuda_ed25519_vanity
```
