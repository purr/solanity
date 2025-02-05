#!/usr/bin/env python3
import sys
import json


def hex_to_keypair(hex_str):
    # Remove '0x' prefix if present
    hex_str = hex_str.replace("0x", "")
    # Convert hex string to bytes and then to list of integers
    bytes_list = [int(hex_str[i : i + 2], 16) for i in range(0, len(hex_str), 2)]
    return bytes_list


def main():
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

    # Get public key (last 32 bytes in base58 format)
    from base58 import b58encode

    pubkey = b58encode(bytes(keypair[32:])).decode("ascii")

    # Create filename with public key
    filename = f"keypair_{pubkey}.json"

    # Save to file
    with open(filename, "w") as f:
        json.dump(keypair, f)

    print(f"Keypair saved to {filename}")


if __name__ == "__main__":
    main()
