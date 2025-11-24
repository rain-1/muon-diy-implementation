import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_BINARY = Path(os.getenv("IMAGE_CLASSIFIER_BIN", BASE_DIR.parent / "build" / "image_classifier"))

app = FastAPI(title="Muon DIY Image Classifier")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

INDEX_HTML = (BASE_DIR / "static" / "index.html").read_text(encoding="utf-8")


def preprocess_image(file: UploadFile) -> bytes:
    try:
        image = Image.open(file.file).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Unable to read image: {exc}")

    resized = image.resize((32, 32))
    array = np.asarray(resized, dtype=np.float32) / 255.0
    chw = np.transpose(array, (2, 0, 1)).flatten()
    return chw.tobytes()


def run_binary(payload: bytes, binary_path: Path = DEFAULT_BINARY) -> List[dict]:
    if not binary_path.exists():
        raise HTTPException(status_code=500, detail=f"Classifier binary not found at {binary_path}")

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        tmp.write(payload)
        tmp_path = Path(tmp.name)

    try:
        result = subprocess.run(
            [str(binary_path), str(tmp_path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except subprocess.CalledProcessError as exc:
        raise HTTPException(status_code=500, detail=f"Classifier failed: {exc.stderr.strip()}") from exc
    except subprocess.TimeoutExpired as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=504, detail="Classifier timed out") from exc
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass

    try:
        parsed = json.loads(result.stdout)
        return parsed.get("labels", [])
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"Invalid classifier output: {exc}") from exc


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return INDEX_HTML


@app.post("/api/classify")
async def classify(file: UploadFile = File(...)) -> dict:
    payload = preprocess_image(file)
    labels = run_binary(payload)
    if not labels:
        raise HTTPException(status_code=500, detail="Classifier returned no labels")
    top_three = labels[:3]
    return {"labels": top_three}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
