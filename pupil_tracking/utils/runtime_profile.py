from __future__ import annotations

import ctypes
import os
import platform
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import cv2


@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    logical_cores: int
    physical_cores: int
    ram_gb: float
    has_cuda: bool
    recommended_resolution: int
    recommended_target_fps: float
    recommended_fp16: bool
    recommended_compile: bool
    recommended_batch_size: int
    recommended_capture_buffer: int
    cv_threads: int
    torch_threads: int
    torch_interop_threads: int


def _detect_total_ram_gb() -> float:
    if platform.system() == "Windows":
        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return float(status.ullTotalPhys) / (1024.0 ** 3)

    page_size = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else None
    phys_pages = os.sysconf("SC_PHYS_PAGES") if hasattr(os, "sysconf") else None
    if page_size and phys_pages:
        return float(page_size * phys_pages) / (1024.0 ** 3)
    return 8.0


def _detect_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


@lru_cache(maxsize=1)
def detect_runtime_profile() -> RuntimeProfile:
    logical = max(1, os.cpu_count() or 1)
    physical = max(1, logical // 2)
    ram_gb = _detect_total_ram_gb()
    has_cuda = _detect_cuda_available()

    if has_cuda:
        return RuntimeProfile(
            name="gpu_accelerated",
            logical_cores=logical,
            physical_cores=physical,
            ram_gb=ram_gb,
            has_cuda=True,
            recommended_resolution=320,
            recommended_target_fps=24.0,
            recommended_fp16=True,
            recommended_compile=True,
            recommended_batch_size=4,
            recommended_capture_buffer=3,
            cv_threads=min(physical, 2),
            torch_threads=min(physical, 6),
            torch_interop_threads=1,
        )

    if ram_gb <= 8.5 or physical <= 4:
        return RuntimeProfile(
            name="cpu_compact",
            logical_cores=logical,
            physical_cores=physical,
            ram_gb=ram_gb,
            has_cuda=False,
            recommended_resolution=320,
            recommended_target_fps=18.0,
            recommended_fp16=False,
            recommended_compile=False,
            recommended_batch_size=1,
            recommended_capture_buffer=2,
            cv_threads=min(physical, 2),
            torch_threads=min(physical, 4),
            torch_interop_threads=1,
        )

    return RuntimeProfile(
        name="cpu_balanced",
        logical_cores=logical,
        physical_cores=physical,
        ram_gb=ram_gb,
        has_cuda=False,
        recommended_resolution=320,
        recommended_target_fps=20.0,
        recommended_fp16=False,
        recommended_compile=False,
        recommended_batch_size=1,
        recommended_capture_buffer=3,
        cv_threads=min(physical, 3),
        torch_threads=min(physical, 6),
        torch_interop_threads=1,
    )


def apply_runtime_optimizations(
    profile: Optional[RuntimeProfile] = None,
) -> RuntimeProfile:
    profile = profile or detect_runtime_profile()

    try:
        cv2.setNumThreads(max(1, int(profile.cv_threads)))
    except Exception:
        pass

    try:
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass

    try:
        import torch

        torch.set_num_threads(max(1, int(profile.torch_threads)))
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(max(1, int(profile.torch_interop_threads)))
    except Exception:
        pass

    return profile
