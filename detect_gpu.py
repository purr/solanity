#!/usr/bin/env python3
import re
import sys
import subprocess


def get_gpu_info():
    try:
        # Run nvidia-smi to get GPU info
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=gpu_name,gpu_compute_capability",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        # Parse the output
        gpu_info = result.stdout.strip().split(",")
        if len(gpu_info) < 2:
            raise ValueError("Couldn't parse GPU information")

        gpu_name = gpu_info[0].strip()
        compute_capability = gpu_info[1].strip()

        # Convert compute capability to sm version (e.g., "8.6" -> "sm_86")
        major, minor = compute_capability.split(".")
        sm_version = f"sm_{major}{minor}"
        compute_version = f"compute_{major}{minor}"

        return gpu_name, sm_version, compute_version

    except subprocess.CalledProcessError:
        print(
            "Error: Could not detect NVIDIA GPU. Make sure nvidia-smi is installed and a CUDA-capable GPU is present."
        )
        sys.exit(1)
    except Exception as e:
        print(f"Error detecting GPU: {str(e)}")
        sys.exit(1)


def update_gpu_common_mk(sm_version, compute_version):
    try:
        # Read the current file
        with open("src/gpu-common.mk", "r") as f:
            content = f.read()

        # Update the architecture settings
        content = re.sub(
            r"GPU_PTX_ARCH:=compute_\d+", f"GPU_PTX_ARCH:={compute_version}", content
        )
        content = re.sub(r"GPU_ARCHS\?=.*", f"GPU_ARCHS?={sm_version}", content)

        # Write the updated content
        with open("src/gpu-common.mk", "w") as f:
            f.write(content)

        return True
    except Exception as e:
        print(f"Error updating gpu-common.mk: {str(e)}")
        return False


def main():
    print("Detecting GPU architecture...")
    gpu_name, sm_version, compute_version = get_gpu_info()
    print(f"Found GPU: {gpu_name}")
    print(f"Compute capability: {sm_version}")

    if update_gpu_common_mk(sm_version, compute_version):
        print(f"Successfully updated src/gpu-common.mk with {sm_version} architecture")
    else:
        print("Failed to update src/gpu-common.mk")


if __name__ == "__main__":
    main()
