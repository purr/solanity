# Solana Vanity Address Generator

A high-performance CUDA-based vanity address generator for Solana. Generate ed25519 keypairs with custom prefixes and suffixes.

## No GPU? Rent One!

You can rent a GPU from vast.ai to run this tool. Use our [referral link](https://cloud.vast.ai/?ref_id=195874) to get started.

### Quick Setup on vast.ai:

1. Visit [our template](<https://cloud.vast.ai/?ref_id=195874&creator_id=195874&name=NVIDIA%20CUDA%20(Ubuntu)%20edited>)
2. Choose a machine (faster GPU = faster results)
3. Connect via SSH when ready
4. Run these commands:

```bash
# Clone and setup
git clone https://github.com/purr/solanity.git
cd solanity
chmod +x mk vanity detect_gpu.py
pip install base58 pynacl

# Build and run
./mk
./vanity -p YourPrefix -s YourSuffix
```

## Prerequisites

- NVIDIA GPU
- CUDA toolkit installed
- nvidia-smi available in PATH
- Python 3.x with pip

## Quick Start

1. **Setup**

```bash
# Clone the repository
git clone https://github.com/yourusername/solanity.git
cd solanity

# Make scripts executable
chmod +x mk vanity detect_gpu.py

# Install Python dependencies
pip install base58 pynacl
```

2. **Build**

```bash
./mk   # This will automatically detect your GPU and build the project
```

3. **Run**

```bash
# Basic usage
./vanity -p ABC        # Find address starting with ABC
./vanity -s XYZ        # Find address ending with XYZ

# Advanced patterns
./vanity -p ABC -p DEF  # Match either prefix ABC or DEF
./vanity -s XYZ -s 123  # Match either suffix XYZ or 123
./vanity -p ABC -s XYZ  # Match both prefix ABC AND suffix XYZ
./vanity -p A?C        # Use ? as wildcard for any character
```

## Command Options

- `-p, --prefix PATTERN` - Add prefix pattern (can use multiple times, up to 32 patterns)
- `-s, --suffix PATTERN` - Add suffix pattern (can use multiple times, up to 32 patterns)
- `-n, --num-keys NUM` - Stop after finding NUM keys (default: 100)
- `-i, --iterations NUM` - Maximum GPU kernel launches (default: 1000000)
- `-a, --attempts NUM` - Attempts per GPU thread (default: 1000000)

## Case Sensitivity and Pattern Matching

The vanity address generator is case-sensitive. To search for all case variations, you can:

1. Use our helper script:

```bash
# Generate all case combinations
python case_combinations.py meow p  # for prefix
python case_combinations.py pump s  # for suffix

# Or use interactive mode:
python case_combinations.py
```

2. Manually specify each variation:

```bash
./vanity -p PUMP -p Pump -p pump  # Different cases
./vanity -p P?MP                  # ? matches any character
```

The program supports up to 32 patterns of each type (prefix/suffix), allowing you to search for all case variations of longer patterns. For example, a 4-letter pattern like "PUMP" has 16 possible case combinations, all of which can be searched simultaneously.

## Understanding the Output

When a match is found, you'll see:

```
MATCH:PublicKey,PrivateKeySeed
```

Example:

```
MATCH:ABC123XYZ...,1234567890abcdef...
```

## Converting Matches to Solana Keypairs

Use the provided conversion script:

```bash
# Option 1: Interactive mode
python convert.py

# Option 2: Direct conversion
python convert.py <hex_string>  # Use the part after the comma from MATCH
```

The script will:

1. Create a Solana keypair from the seed
2. Save it in the `keypairs` directory as `keypair_PUBLICKEY.json`
3. Display the public key for verification

Your keypair files will be stored in the `keypairs` directory, which is automatically created and git-ignored for security.

## Performance

Performance varies by GPU model and pattern complexity:

- Simple patterns (e.g., "ABC"): seconds to minutes
- Complex patterns (e.g., "ABCDE"): minutes to hours
- Multiple patterns: minimal performance impact

The program automatically utilizes all available CUDA-capable GPUs in your system. Each GPU runs in parallel, significantly increasing the search speed. At startup, you'll see detailed information about each GPU being used, including:

- GPU model name
- Compute capability
- Available memory
- Number of CUDA cores
- Current thread configuration

## GPU Support

The project automatically detects and configures for your GPU:

| GPU Series     | Compute Capability |
| -------------- | ------------------ |
| RTX 50 Series  | 9.0                |
| RTX 40 Series  | 8.9                |
| RTX 30 Series  | 8.6                |
| RTX 20 Series  | 7.5                |
| GTX 16 Series  | 7.5                |
| GTX 10 Series  | 6.1                |
| GTX 900 Series | 5.2                |

## Important Notes

1. Case-sensitive matching ("ABC" won't match "abc")
2. Use ? as wildcard for single character
3. Not recommended for production use (uses CURAND's XORWOW algorithm)
4. No warranty or liability - use at your own risk

## Troubleshooting

1. **GPU Detection Fails**

   - Ensure CUDA toolkit is installed
   - Check if nvidia-smi works
   - GPU architecture can be set manually in `src/gpu-common.mk`

2. **Build Errors**

   - Make sure CUDA is in your PATH
   - Try: `export PATH=/usr/local/cuda/bin:$PATH`

3. **Runtime Errors**
   - Check library path: `export LD_LIBRARY_PATH=./src/release:$LD_LIBRARY_PATH`
   - Verify GPU compute capability is correct

## Credits

Originally forked from https://github.com/ChorusOne/solanity with improvements:

- Entropy-based initialization
- Case-sensitive pattern matching
- Configurable exit criteria
- Automatic GPU detection
- Improved keypair handling
