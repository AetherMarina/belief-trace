import sys
import subprocess


def check_environment():
    print("=" * 50)
    print("BELIEF-TRACE: ENVIRONMENT DIAGNOSTICS")
    print("=" * 50)

    # 1. System & Python Info
    print("\n[1] SYSTEM & PYTHON")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"Python Executable: {sys.executable}")

    # 2. PyTorch & CUDA
    print("\n[2] PYTORCH & CUDA")
    try:
        import torch
        print(f"PyTorch Location: {torch.__file__}")
        print(f"PyTorch Version:  {torch.__version__}")
        print(f"CUDA Compiled:    {torch.version.cuda}")
        print(f"CUDA Available:   {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"GPU Count:        {torch.cuda.device_count()}")
            print(f"Active GPU:       {torch.cuda.get_device_name(0)}")
        else:
            print("\nWARNING: PyTorch cannot find a CUDA-enabled GPU.")
            print("1. You don't have an NVIDIA GPU.")
            print("2. Your NVIDIA drivers are not installed correctly.")
            print("3. You installed a CPU-only version of PyTorch.")
    except ImportError:
        print("PyTorch is NOT installed.")

    # 3. Critical Dependencies (Unsloth / LLM Stack)
    print("\n[3] LLM DEPENDENCIES")

    # Check xformers
    try:
        import xformers
        print(f"xformers:      {xformers.__version__}")
    except ImportError:
        print("xformers:      NOT INSTALLED")

    # Check bitsandbytes
    try:
        import bitsandbytes as bnb
        print(f"bitsandbytes:  {bnb.__version__}")
    except ImportError:
        print("bitsandbytes:  NOT INSTALLED")

    print("=" * 50)


if __name__ == "__main__":
    check_environment()
