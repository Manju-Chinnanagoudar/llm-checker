import platform
import psutil
import cpuinfo

from dataclasses import dataclass
from typing import Optional


@dataclass
class GPUInfo:
    name: str
    vram_gb: float
    backend: str  # cuda, metal, rocm, none


@dataclass
class HardwareProfile:
    # RAM
    total_ram_gb: float
    available_ram_gb: float

    # CPU
    cpu_name: str
    cpu_cores: int
    has_avx: bool
    has_avx2: bool
    has_avx512: bool

    # GPU
    gpus: list[GPUInfo]

    # Disk
    free_disk_gb: float

    # OS
    os: str  # windows, linux, macos


def _get_gpu_info() -> list[GPUInfo]:
    gpus = []

    # --- NVIDIA (CUDA) ---
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            vram_gb = round(mem.total / (1024 ** 3), 2)
            gpus.append(GPUInfo(name=name, vram_gb=vram_gb, backend="cuda"))
        pynvml.nvmlShutdown()
        return gpus
    except Exception:
        pass

    # --- Apple Metal ---
    if platform.system() == "Darwin":
        try:
            import subprocess
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True, text=True, timeout=10
            )
            lines = result.stdout
            if "Metal" in lines:
                # Extract VRAM if available
                vram_gb = 0.0
                for line in lines.splitlines():
                    if "VRAM" in line:
                        parts = line.split(":")
                        if len(parts) > 1:
                            val = parts[1].strip().lower()
                            if "gb" in val:
                                try:
                                    vram_gb = float(val.replace("gb", "").strip())
                                except ValueError:
                                    pass
                gpus.append(GPUInfo(
                    name="Apple Silicon GPU",
                    vram_gb=vram_gb,
                    backend="metal"
                ))
            return gpus
        except Exception:
            pass

    # --- AMD (ROCm) ---
    try:
        import subprocess
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            import json
            data = json.loads(result.stdout)
            for card, info in data.items():
                vram_bytes = int(info.get("VRAM Total Memory (B)", 0))
                vram_gb = round(vram_bytes / (1024 ** 3), 2)
                gpus.append(GPUInfo(
                    name=f"AMD GPU ({card})",
                    vram_gb=vram_gb,
                    backend="rocm"
                ))
            return gpus
    except Exception:
        pass

    return gpus  # empty = CPU only


def _get_os() -> str:
    system = platform.system()
    if system == "Darwin":
        return "macos"
    elif system == "Windows":
        return "windows"
    return "linux"


def get_hardware_profile() -> HardwareProfile:
    # RAM
    ram = psutil.virtual_memory()
    total_ram_gb = round(ram.total / (1024 ** 3), 2)
    available_ram_gb = round(ram.available / (1024 ** 3), 2)

    # CPU
    info = cpuinfo.get_cpu_info()
    cpu_name = info.get("brand_raw", "Unknown CPU")
    cpu_cores = psutil.cpu_count(logical=False) or 1
    flags = info.get("flags", [])
    has_avx    = "avx"    in flags
    has_avx2   = "avx2"   in flags
    has_avx512 = any(f.startswith("avx512") for f in flags)

    # GPU
    gpus = _get_gpu_info()

    # Disk (root partition)
    disk = psutil.disk_usage("/")
    free_disk_gb = round(disk.free / (1024 ** 3), 2)

    return HardwareProfile(
        total_ram_gb=total_ram_gb,
        available_ram_gb=available_ram_gb,
        cpu_name=cpu_name,
        cpu_cores=cpu_cores,
        has_avx=has_avx,
        has_avx2=has_avx2,
        has_avx512=has_avx512,
        gpus=gpus,
        free_disk_gb=free_disk_gb,
        os=_get_os(),
    )


def display_profile(profile: HardwareProfile) -> None:
    """Pretty print the hardware profile using Rich."""
    from rich.console import Console
    from rich.table import Table
    from rich import box

    console = Console()
    table = Table(box=box.ROUNDED, show_header=False, border_style="cyan")
    table.add_column("Property", style="bold cyan", width=22)
    table.add_column("Value", style="white")

    table.add_row("OS", profile.os.capitalize())
    table.add_row("CPU", profile.cpu_name)
    table.add_row("CPU Cores", str(profile.cpu_cores))
    table.add_row(
        "CPU Features",
        " ".join(filter(None, [
            "AVX"    if profile.has_avx    else "",
            "AVX2"   if profile.has_avx2   else "",
            "AVX512" if profile.has_avx512 else "",
        ])) or "None detected"
    )
    table.add_row("Total RAM", f"{profile.total_ram_gb} GB")
    table.add_row("Available RAM", f"{profile.available_ram_gb} GB")
    table.add_row("Free Disk", f"{profile.free_disk_gb} GB")

    if profile.gpus:
        for i, gpu in enumerate(profile.gpus):
            table.add_row(
                f"GPU {i + 1}",
                f"{gpu.name} — {gpu.vram_gb} GB VRAM ({gpu.backend.upper()})"
            )
    else:
        table.add_row("GPU", "None detected (CPU-only mode)")

    console.print(table)