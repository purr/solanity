#!/usr/bin/env python3
import os
import sys
import json

from base58 import b58encode
from nacl.signing import SigningKey  # type: ignore


def hex_to_keypair(hex_str):
    # Remove '0x' prefix if present
    hex_str = hex_str.replace("0x", "")

    # Validate hex string length (should be 64 bytes = 128 hex chars)
    if len(hex_str) != 64:
        raise ValueError(
            f"Invalid hex string length. Expected 64 bytes (128 hex chars), got {len(hex_str) // 2} bytes"
        )

    try:
        # Convert hex string to bytes and then to list of integers
        seed = [int(hex_str[i : i + 2], 16) for i in range(0, len(hex_str), 2)]

        # For Solana keypair, we need both private and public key parts
        # The seed is the private key, we need to derive the public key

        private_key = bytes(seed)
        signing_key = SigningKey(private_key)
        public_key = list(signing_key.verify_key.encode())

        # Combine private and public key
        keypair = seed + public_key
        return keypair
    except ValueError as e:
        raise ValueError(f"Invalid hex string: {str(e)}")


def main():
    try:
        # Create keypairs directory if it doesn't exist
        os.makedirs("keypairs", exist_ok=True)

        # Get hex input either from command line or user input
        if len(sys.argv) > 1:
            hex_input = sys.argv[1]
        else:
            print(
                "Please enter the hex string from the MATCH line (the part after the comma):"
            )
            hex_input = input().strip()

        # Convert hex to keypair format
        keypair = hex_to_keypair(hex_input)

        # Get public key (last 32 bytes)
        public_key = bytes(keypair[32:])
        pubkey = b58encode(public_key).decode("ascii")

        # Create filename with public key in the keypairs directory
        filename = os.path.join("keypairs", f"keypair_{pubkey}.json")

        # Save to file
        with open(filename, "w") as f:
            json.dump(keypair, f)

        print(f"Keypair saved to {filename}")
        print(f"Public Key: {pubkey}")

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
