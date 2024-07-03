import multiprocessing as mp
import sys
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from functools import partial
from pathlib import Path
import subprocess
import shutil

import typer


app = typer.Typer()


class Task(str, Enum):
    total = "total"
    regions = "regions"


def worker_callback(f):
    e = f.exception()

    if e is None:
        return

    trace = []
    tb = e.__traceback__
    while tb is not None:
        trace.append(
            {
                "filename": tb.tb_frame.f_code.co_filename,
                "name": tb.tb_frame.f_code.co_name,
                "lineno": tb.tb_lineno,
            }
        )
        tb = tb.tb_next
    print(str({"type": type(e).__name__, "message": str(e), "trace": trace}))


@app.command(name="run")
def run_tvs(
    input_path: Path,
    output_path: Path,
    task: Task = Task.total,
    keep_size: bool = False,
    fill_holes: bool = False,
    crop: bool = False,
    copy_tissue_info: bool = True,
):
    """Run TVS"""

    task_map = {Task.total: 87, Task.regions: 278}
    dataset_id = task_map[task]

    args = [
        sys.executable,
        f"{Path(__file__).parent / 'run_TotalVibeSegmentator.py'}",
        "--img",
        f"{input_path}",
        "--out_path",
        f"{output_path}",
        "--dataset_id",
        f"{dataset_id}",
    ]
    if keep_size:
        args.append("--keep_size")
    if fill_holes:
        args.append("--fill_holes")
    if crop:
        args.append("--crop")

    print(" ".join(args))

    with open(output_path.with_suffix(".log"), "w") as logfile:
        logfile.write(f"Running: {' '.join(args)}\n\n")
        prog = subprocess.Popen(
            args,
            stdout=logfile,
            stderr=logfile,
            cwd=str(output_path.parent),
            shell=True,
        )
        ok = prog.communicate()

        if copy_tissue_info:
            tissues_path = Path(__file__).parent / f"tissues_{Task.total}.txt"
            shutil.copyfile(tissues_path, output_path.parent / "tissues.txt")

        return ok


@app.command()
def run_all(
    input_dir: Path,
    output_dir: Path,
    task: Task = Task.total,
    keep_size: bool = False,
    fill_holes: bool = False,
    crop: bool = False,
    glob: str = "*.nii.gz",
    num_processes: int = 2,
):
    """Run TVS on all images in folder"""

    mp.freeze_support()
    files = list(input_dir.glob(glob))

    output_dir.mkdir(exist_ok=True, parents=True)

    tissues_path = Path(__file__).parent / f"tissues_{Task.total}.txt"
    shutil.copyfile(tissues_path, output_dir / "tissues.txt")

    kwargs = {
        "keep_size": keep_size,
        "fill_holes": fill_holes,
        "crop": crop,
        "copy_tissue_info": False,
    }

    tp = ThreadPoolExecutor(max_workers=num_processes)
    for f in files:
        output_file = output_dir / f.name
        tp.submit(
            partial(run_tvs, f, output_file, task=task, **kwargs)
        ).add_done_callback(worker_callback)


if __name__ == "__main__":
    app()
