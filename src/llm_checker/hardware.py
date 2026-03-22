import json
import platform
import subprocess
from dataclasses import dataclass

import cpuinfo
import psutil


@dataclass
class GPUInfo:
    name: str
    vram_gb: float
    backend: str  # cuda | metal | rocm


@dataclass
class HardwareProfile:
    total_ram_gb: float
    available_ram_gb: float
    cpu_name: str
    cpu_cores: int
    has_avx: bool
    has_avx2: bool
    has_avx512: bool
    gpus: list[GPUInfo]
    free_disk_gb: float
    os: str


def _nvidia_gpus() -> list[GPUInfo]:
    try:
        import pynvml
        pynvml.nvmlInit()
        gpus = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            gpus.append(GPUInfo(
                name=pynvml.nvmlDeviceGetName(h),
                vram_gb=round(mem.total / (1024 ** 3), 2),
                backend="cuda",
            ))
        pynvml.nvmlShutdown()
        return gpus
    except Exception:
        return []


def _apple_gpus() -> list[GPUInfo]:
    if platform.system() != "Darwin":
        return []
    try:
        out = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True, text=True, timeout=10
        ).stdout
        if "Metal" not in out:
            return []
        vram_gb = 0.0
        for line in out.splitlines():
            if "VRAM" in line and ":" in line:
                val = line.split(":")[1].strip().lower()
                if "gb" in val:
                    try:
                        vram_gb = float(val.replace("gb", "").strip())
                    except ValueError:
                        pass
        # Apple Silicon uses unified memory — if we couldn't parse VRAM
        # from system_profiler, fall back to total RAM as an approximation
        if vram_gb == 0.0:
            import psutil
            vram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
        return [GPUInfo(name="Apple Silicon GPU", vram_gb=vram_gb, backend="metal")]
    except Exception:
        return []


def _amd_gpus() -> list[GPUInfo]:
    # rocm-smi might not exist, that's fine
    try:
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return []
        gpus = []
        for card, info in json.loads(result.stdout).items():
            vram_bytes = int(info.get("VRAM Total Memory (B)", 0))
            gpus.append(GPUInfo(
                name=f"AMD GPU ({card})",
                vram_gb=round(vram_bytes / (1024 ** 3), 2),
                backend="rocm",
            ))
        return gpus
    except Exception:
        return []


def _detect_gpus() -> list[GPUInfo]:
    # nvidia first since it's the most common, fall through to others
    gpus = _nvidia_gpus()
    if gpus:
        return gpus
    gpus = _apple_gpus()
    if gpus:
        return gpus
    return _amd_gpus()


def _get_os() -> str:
    s = platform.system()
    if s == "Darwin":
        return "macos"
    if s == "Windows":
        return "windows"
    return "linux"


def get_hardware_profile() -> HardwareProfile:
    ram = psutil.virtual_memory()
    cpu_info = cpuinfo.get_cpu_info()
    flags = cpu_info.get("flags", [])

    # TODO: handle edge cases where / isn't the main partition
    disk_path = "C:\\" if _get_os() == "windows" else "/"
    disk = psutil.disk_usage(disk_path)

    return HardwareProfile(
        total_ram_gb=round(ram.total / (1024 ** 3), 2),
        available_ram_gb=round(ram.available / (1024 ** 3), 2),
        cpu_name=cpu_info.get("brand_raw", "Unknown CPU"),
        cpu_cores=psutil.cpu_count(logical=False) or 1,
        has_avx="avx" in flags,
        has_avx2="avx2" in flags,
        has_avx512=any(f.startswith("avx512") for f in flags),
        gpus=_detect_gpus(),
        free_disk_gb=round(disk.free / (1024 ** 3), 2),
        os=_get_os(),
    )


def display_profile(profile: HardwareProfile) -> None:
    from rich import box
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(box=box.ROUNDED, show_header=False, border_style="cyan")
    table.add_column("", style="bold cyan", width=22)
    table.add_column("", style="white")

    avx_flags = " ".join(filter(None, [
        "AVX" if profile.has_avx else "",
        "AVX2" if profile.has_avx2 else "",
        "AVX512" if profile.has_avx512 else "",
    ])) or "none"

    table.add_row("OS", profile.os.capitalize())
    table.add_row("CPU", profile.cpu_name)
    table.add_row("Cores", str(profile.cpu_cores))
    table.add_row("CPU extensions", avx_flags)
    table.add_row("Total RAM", f"{profile.total_ram_gb} GB")
    table.add_row("Available RAM", f"{profile.available_ram_gb} GB")
    table.add_row("Free disk", f"{profile.free_disk_gb} GB")

    if profile.gpus:
        for i, g in enumerate(profile.gpus):
            table.add_row(f"GPU {i + 1}", f"{g.name} — {g.vram_gb} GB ({g.backend.upper()})")
    else:
        table.add_row("GPU", "none (CPU-only)")

    console.print(table)