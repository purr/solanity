#!/usr/bin/env python3
import re
import sys
import subprocess


def get_gpu_info():
    try:
        # Run nvidia-smi to get GPU name
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=gpu_name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )

        gpu_name = result.stdout.strip().lower()
        print(f"Found GPU: {gpu_name}")

        # Map GPU series to compute capabilities

        # RTX 50 series (Blackwell) - Compute 9.0
        if gpu_name.find("50") != -1 and any(
            x in gpu_name for x in ["5090", "5080", "5070", "5060"]
        ):
            print("RTX 50 series detected, using compute_90")
            return gpu_name, "sm_90", "compute_90"

        # RTX 40 series (Ada Lovelace) - Compute 8.9
        if gpu_name.find("40") != -1 and any(
            x in gpu_name for x in ["4090", "4080", "4070", "4060"]
        ):
            print("RTX 40 series detected, using compute_89")
            return gpu_name, "sm_89", "compute_89"

        # RTX 30 series (Ampere) - Compute 8.6
        if gpu_name.find("30") != -1 and any(
            x in gpu_name for x in ["3090", "3080", "3070", "3060", "3050"]
        ):
            print("RTX 30 series detected, using compute_86")
            return gpu_name, "sm_86", "compute_86"

        # RTX 20 series (Turing) - Compute 7.5
        if gpu_name.find("20") != -1 and any(
            x in gpu_name for x in ["2080", "2070", "2060"]
        ):
            print("RTX 20 series detected, using compute_75")
            return gpu_name, "sm_75", "compute_75"

        # GTX 16 series (Turing) - Compute 7.5
        if gpu_name.find("16") != -1 and any(x in gpu_name for x in ["1660", "1650"]):
            print("GTX 16 series detected, using compute_75")
            return gpu_name, "sm_75", "compute_75"

        # GTX 10 series - Compute 6.1
        if gpu_name.find("10") != -1 and any(
            x in gpu_name for x in ["1080", "1070", "1060", "1050"]
        ):
            print("GTX 10 series detected, using compute_61")
            return gpu_name, "sm_61", "compute_61"

        # Professional GPUs (Quadro/RTX Professional)
        if "quadro" in gpu_name or "rtx" in gpu_name:
            print("Professional GPU detected, using compute_86")
            return gpu_name, "sm_86", "compute_86"

        # Fallback to safe default
        print("Could not determine exact GPU model, using compute_86 as safe default")
        return gpu_name, "sm_86", "compute_86"

    except subprocess.CalledProcessError as e:
        print(
            "Error: Could not detect NVIDIA GPU. Make sure nvidia-smi is installed and a CUDA-capable GPU is present."
        )
        print(f"nvidia-smi error: {e.stderr}")

        # Try nvcc as fallback
        try:
            result = subprocess.run(
                ["nvcc", "--version"], capture_output=True, text=True, check=True
            )
            print("Found CUDA installation, using compute_86 as safe default")
            return "Unknown", "sm_86", "compute_86"
        except Exception:
            print(
                "Could not detect GPU at all. Please edit src/gpu-common.mk manually."
            )
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
    print(f"Compute capability: {sm_version}")

    if update_gpu_common_mk(sm_version, compute_version):
        print(f"Successfully updated src/gpu-common.mk with {sm_version} architecture")
    else:
        print("Failed to update src/gpu-common.mk")


if __name__ == "__main__":
    main()
