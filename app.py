"""
ATM Captcha Resolver — Web Service
Expone un endpoint POST /resolve que recibe una imagen y retorna el texto del captcha.
Requiere: model/captcha_model.onnx y model/vocab.json junto al script.
"""

import os
import sys
import json
import tempfile

import numpy as np
import cv2
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

IMG_W, IMG_H = 128, 48

def resource_path(relative_path: str) -> str:
    if hasattr(sys, "_MEIPASS"):
        base_path = sys._MEIPASS  # type: ignore
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)

MODEL_PATH = resource_path("model/captcha_model.onnx")
VOCAB_PATH  = resource_path("model/vocab.json")

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modelo no encontrado: {MODEL_PATH}")
    if not os.path.exists(VOCAB_PATH):
        raise FileNotFoundError(f"Vocabulario no encontrado: {VOCAB_PATH}")
    sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    return sess, vocab

def preprocess(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"No se pudo leer la imagen: {image_path}")
    hsv   = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask  = (hsv[:, :, 1] > 30).astype(np.uint8)
    clean = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    gray  = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
    gray  = cv2.resize(gray, (IMG_W, IMG_H), interpolation=cv2.INTER_CUBIC)
    return gray.astype(np.float32) / 255.0

def decode(logits: np.ndarray, vocab: list) -> str:
    preds = np.argmax(logits[:, 0, :], axis=1)
    chars, prev = [], -1
    for p in preds:
        if p != prev and p != 0:
            chars.append(vocab[int(p)])
        prev = p
    return "".join(chars)

def recognize(image_path: str, sess, vocab) -> str:
    img    = preprocess(image_path)
    inp    = img[None, None, :, :]
    logits = sess.run(["logits"], {"image": inp})[0]
    return decode(logits, vocab).upper()


app = FastAPI(title="ATM Captcha Resolver", version="1.0.0")

try:
    _sess, _vocab = load_model()
except Exception as e:
    _sess, _vocab = None, None
    print(f"[WARN] No se pudo cargar el modelo al arrancar: {e}", file=sys.stderr)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _sess is not None}


@app.post("/resolve")
async def resolve(image: UploadFile = File(...)):
    if _sess is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    allowed = {"image/png", "image/jpeg", "image/jpg", "image/bmp", "image/webp"}
    if image.content_type not in allowed:
        raise HTTPException(status_code=400, detail=f"Tipo de archivo no soportado: {image.content_type}")

    suffix = os.path.splitext(image.filename or "img.png")[-1] or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await image.read())
        tmp_path = tmp.name

    try:
        result = recognize(tmp_path, _sess, _vocab)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    finally:
        os.unlink(tmp_path)

    return JSONResponse({"captcha": result})
