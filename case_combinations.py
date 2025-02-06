import sys
from itertools import product


def generate_case_combinations(input_str):
    """Generate all possible case combinations for a given string."""
    # Get all possible combinations of upper/lower case for each character
    cases = product(*[(c.lower(), c.upper()) for c in input_str])
    return ["".join(case) for case in cases]


def format_command(combinations, pattern_type):
    """Format the combinations into a vanity command string."""
    # Determine the flag based on pattern type
    flag = "-s" if pattern_type.lower().startswith("s") else "-p"

    # Join all combinations with the appropriate flag
    return " ".join(f"{flag} {combo}" for combo in combinations)


def get_user_input():
    """Get pattern and type interactively from user."""
    print("\nEnter the pattern you want to search for:")
    pattern = input("> ").strip()

    while not pattern:
        print("Pattern cannot be empty. Please try again:")
        pattern = input("> ").strip()

    print("\nSelect pattern type:")
    print("1. Suffix (-s)")
    print("2. Prefix (-p)")

    while True:
        choice = input("Enter 1 or 2: ").strip()
        if choice == "1":
            return pattern, "s"
        elif choice == "2":
            return pattern, "p"
        print("Invalid choice. Please enter 1 for suffix or 2 for prefix.")


def main():
    # Check if arguments were provided via command line
    if len(sys.argv) == 3:
        input_str = sys.argv[1]
        pattern_type = sys.argv[2]

        if pattern_type.lower() not in ["s", "p"]:
            print("Error: Second argument must be 's' for suffix or 'p' for prefix")
            sys.exit(1)
    else:
        # If not enough arguments, use interactive mode
        if len(sys.argv) > 1:
            print("Usage: python case_combinations.py <string> <s|p>")
            print("  s: suffix mode")
            print("  p: prefix mode")
            print("\nExample:")
            print("  python case_combinations.py meow s")
            print("  python case_combinations.py ABC p")
            print("\nNo arguments provided - switching to interactive mode...")

        input_str, pattern_type = get_user_input()

    # Generate all possible combinations
    combinations = generate_case_combinations(input_str)

    # Check if we exceed the pattern limit
    MAX_PATTERNS = 32  # This should match the value in config.h
    if len(combinations) > MAX_PATTERNS:
        print(
            f"\nWarning: Generated {len(combinations)} combinations, but the program only supports {MAX_PATTERNS} patterns."
        )
        print(
            "The command below will only use the first 32 patterns. You may need to run multiple searches."
        )
        print(
            "Consider using a shorter pattern or splitting your search into multiple runs."
        )

    # Format and print the command
    command = format_command(combinations, pattern_type)
    print("\nGenerated command:")
    print(command)

    # Print statistics
    print(f"\nTotal combinations: {len(combinations)}")
    print("\nAll combinations:")
    for i, combo in enumerate(combinations, 1):
        print(f"{i}. {combo}")


if __name__ == "__main__":
    main()
