"""Container-aware CPU facts: what os.cpu_count() reports vs what this container may use.

Containers limit CPU two ways, and os.cpu_count() sees neither:
  - a cpuset (WHICH cpus)      -> os.sched_getaffinity(0)
  - a CFS quota (HOW MUCH time) -> /sys/fs/cgroup/cpu.max  (cgroup v2) or cpu.cfs_quota_us (v1)
On our RunPod pod: os.cpu_count() = 96, affinity = 96, quota = 8.5 cores.
"""
import math
import os
import subprocess


def cgroup_quota():
    """CPU quota in cores (e.g. 8.5), or None when unlimited or not on Linux."""
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            quota, period = f.read().split()
        if quota != "max":
            return int(quota) / int(period)
    except (OSError, ValueError):
        pass
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
            q = int(f.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
            p = int(f.read())
        if q > 0:
            return q / p
    except (OSError, ValueError):
        pass
    return None


def affinity_cpus():
    """CPUs this process may run on (cpuset / taskset), or None on macOS."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return None


def effective_cores():
    """The smallest of: quota (floored), affinity, os.cpu_count()."""
    q = cgroup_quota()
    candidates = [c for c in (math.floor(q) if q else None, affinity_cpus(), os.cpu_count()) if c]
    return max(1, min(candidates))


def describe():
    return (f"os.cpu_count={os.cpu_count()} affinity={affinity_cpus()} "
            f"quota={cgroup_quota()} -> effective={effective_cores()}")


def throttle_stats():
    """(nr_throttled, throttled_ms) for this container's cgroup, or None.
    Note: counts the whole container, so a load generator in the same pod is included."""
    for path, unit in (("/sys/fs/cgroup/cpu.stat", "throttled_usec"),
                       ("/sys/fs/cgroup/cpu/cpu.stat", "throttled_time")):
        try:
            with open(path) as f:
                d = dict(line.split() for line in f if line.strip())
            ms = int(d.get(unit, 0)) / (1e3 if unit == "throttled_usec" else 1e6)
            return int(d.get("nr_throttled", 0)), ms
        except (OSError, ValueError):
            continue
    return None


def thread_count(pid):
    """OS threads in a process: /proc on Linux, `ps -M` on macOS."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("Threads:"):
                    return int(line.split()[1])
    except OSError:
        pass
    try:
        out = subprocess.run(["ps", "-M", "-p", str(pid)], capture_output=True,
                             text=True, check=False).stdout
        return max(0, len(out.strip().splitlines()) - 1)
    except OSError:
        return None
