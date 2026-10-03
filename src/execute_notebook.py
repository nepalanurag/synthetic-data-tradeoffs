"""Minimal notebook executor: runs every code cell of an .ipynb top to bottom
in one shared namespace, captures stdout/stderr as stream outputs and
exceptions as error outputs, then writes the notebook back with outputs.

Used because jupyter/nbclient are not installed in this environment.
Behavior matches nbconvert --execute for plain print-based notebooks.
"""

import io
import sys
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr

import nbformat


def execute(path_in, path_out=None, timeout=600):
    nb = nbformat.read(path_in, as_version=4)
    ns = {}
    path_out = path_out or path_in
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        t0 = time.time()
        buf_out, buf_err = io.StringIO(), io.StringIO()
        outputs = []
        try:
            with redirect_stdout(buf_out), redirect_stderr(buf_err):
                exec(compile(cell.source, f"<cell {i}>", "exec"), ns)
        except Exception:
            traceback.print_exc(file=buf_err)
            outputs.append(
                nbformat.v4.new_output(
                    "error",
                    ename=sys.exc_info()[0].__name__,
                    evalue=str(sys.exc_info()[1]),
                    traceback=traceback.format_exception(*sys.exc_info()),
                )
            )
        if buf_out.getvalue():
            outputs.append(nbformat.v4.new_output("stream", name="stdout",
                                                   text=buf_out.getvalue()))
        if buf_err.getvalue():
            outputs.append(nbformat.v4.new_output("stream", name="stderr",
                                                   text=buf_err.getvalue()))
        cell.outputs = outputs
        cell.execution_count = i + 1
        print(f"cell {i}: {time.time()-t0:.1f}s "
              f"{'ERROR' if any(o.output_type=='error' for o in outputs) else 'ok'}",
              flush=True)
        if any(o.output_type == "error" for o in outputs):
            nbformat.write(nb, path_out)
            raise RuntimeError(f"cell {i} raised; notebook left with partial outputs")
    nbformat.write(nb, path_out)
    print("notebook executed and saved:", path_out, flush=True)


if __name__ == "__main__":
    execute(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
