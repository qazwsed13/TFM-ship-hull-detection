# -*- coding: utf-8 -*-
"""环境自检: 打印复现所需的关键依赖版本与硬件信息."""
import platform
import sys


def main():
    print("=" * 60)
    print("方向2 仿真环境自检")
    print("=" * 60)
    print(f"OS        : {platform.system()} {platform.release()} ({platform.version()})")
    print(f"Python    : {sys.version.split()[0]}")
    for mod in ["numpy", "torch", "torchvision", "cv2", "scipy", "matplotlib"]:
        try:
            m = __import__(mod)
            print(f"{mod:10s}: {getattr(m, '__version__', '未知')}")
        except ImportError:
            print(f"{mod:10s}: 未安装")
    try:
        import torch
        print(f"CUDA      : {torch.version.cuda} | 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU       : {torch.cuda.get_device_name(0)} "
                  f"({torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB)")
    except Exception as e:
        print("CUDA 检查失败:", e)
    print("=" * 60)


if __name__ == "__main__":
    main()
