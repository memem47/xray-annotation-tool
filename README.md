# X‑ray Annotation Tool

An interactive desktop application written in **Python/Tkinter** that lets you:

* **Load real X‑ray images** (PNG/JPEG/TIFF)
* **Generate synthetic X‑ray‑like phantoms** for quick testing when real data are unavailable
* **Draw polygon annotations with the mouse** (left‑click = add vertex, right‑click = close)
* **Export the annotation as a binary mask (PNG)** and save the raw point list as JSON

## Quick Start

```bash
# clone the repository
git clone https://github.com/<your‑username>/xray-annotation-tool.git
cd xray-annotation-tool

# set up isolated environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# install runtime deps
pip install -r requirements.txt

# launch
python src/main.py