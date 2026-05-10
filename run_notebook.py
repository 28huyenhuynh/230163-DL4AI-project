"""
Run the project notebook with Windows-compatible event loop policy.

Without this, nbconvert / ZMQ fails on Windows 10 with:
  zmq.error.ZMQError: not a socket
  RuntimeWarning: Proactor event loop does not implement add_reader...
"""
import asyncio
# Must be set BEFORE any ZMQ / tornado import
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import nbformat
from nbclient import NotebookClient
import os, sys

NB = "230163_project_notebook.ipynb"

print(f"Loading {NB} ...")
with open(NB, encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

print(f"Cells before execution: {len(nb.cells)}")

# Mark the pip-install cell to be skipped (it fails and is not needed since
# packages are already installed in the Python 3.11 environment).
for cell in nb.cells:
    src = "".join(cell.get("source", []))
    if cell["cell_type"] == "code" and "!pip install" in src:
        # Replace with a no-op so the kernel doesn't try to install
        cell["source"] = "print('pip install cell skipped (packages already installed)')"
        print("  → pip-install cell replaced with no-op")

client = NotebookClient(
    nb,
    timeout=7200,           # 2 h per cell max
    kernel_name="python3",
    force_raise_errors=False,   # record errors in cell outputs but keep going
    resources={"metadata": {"path": os.path.dirname(os.path.abspath(NB))}},
)

print("Executing notebook (this will take ~60-90 min on CPU) ...")
try:
    client.execute()
    print("Execution complete.")
except Exception as e:
    print(f"Execution stopped with error: {e}")

print(f"Saving {NB} ...")
with open(NB, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)
print("Saved.")
