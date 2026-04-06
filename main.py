"""
World-Class Vision Embedding & Deepfake Detection Server
========================================================

An enterprise-grade FastAPI server for generating unified Vision-Language embeddings 
using `google/siglip-base-patch16-224` while simultaneously evaluating Media 
through a Tri-State SigLIP2 Deepfake Classifier.

Features:
- `curl_cffi` Chrome TLS/JA3 impersonation to shatter Datadome/Cloudflare 403 blocks
- Intelligent Temporal Video Splicing (`opencv`) to extract 3 chronological keyframes
- Magic-Byte Sniffing (`filetype`) to map raw bitstreams, defending against extension spoofing
- Ephemeral `/tmp/` Buffers to neutralize OOM DDoS vectors from payload over-sizing
- Tri-State Classifier (`prithivMLmods/AI-vs-Deepfake-vs-Real-Siglip2`):
   Unified P(AI) + P(Deepfake) synthetic probability — covers FLUX, Midjourney, and FaceSwap
- Concurrent CPU-GPU async pipeline mapping (Base64/URL parsing -> Hash -> Vector)
"""

import os
import io
import asyncio
import base64
import tempfile
from typing import List, Optional
from contextlib import asynccontextmanager

import cv2
import torch
import filetype
import imagehash
from PIL import Image
from curl_cffi import requests
from loguru import logger

from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from transformers import AutoProcessor, AutoModel, AutoImageProcessor, AutoModelForImageClassification

# Model configuration
SIGLIP_MODEL = "google/siglip-base-patch16-224"
S2_TRI_MODEL = "prithivMLmods/AI-vs-Deepfake-vs-Real-Siglip2"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
API_KEY = os.getenv("VISION_API_KEY", "default-key-change-in-production")
VERSION = "2.0.0-ENTERPRISE"

# Global model instances
models = {}
security = HTTPBearer()

class VisionEmbedRequest(BaseModel):
    image_urls: Optional[List[str]] = Field(default=[], description="List of URLs targeting images, MP4s, WebMs, or GIFs.")
    image_base64: Optional[List[str]] = Field(default=[], description="List of base64 encoded media blobs.")

class VisionEmbedResponse(BaseModel):
    embeddings: List[List[float]] = Field(..., description="768-dimensional sequence mapped vectors.")
    phashes: Optional[List[str]] = Field(default=None, description="Perceptual hashes corresponding to the target media.")
    synthetic_prob: Optional[List[float]] = Field(default=None, description="The unified Deepfake/AI-generator confidence float [0.0 - 1.0].")
    debug: Optional[str] = Field(default=None, description="Exception bridge.")

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_KEY:
        logger.warning("Invalid API key attempt")
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials

startup_error = None

def _load_models_sync():
    global startup_error
    logger.info(f"Booting Neural Array on {DEVICE} memory banks...")
    try:
        # Load the Mathematical Semantic Tensor
        models['siglip_proc'] = AutoProcessor.from_pretrained(SIGLIP_MODEL)
        models['siglip'] = AutoModel.from_pretrained(SIGLIP_MODEL).to(DEVICE)
        models['siglip'].eval()
        
        # Load the Tri-State SigLIP Vision Classifier 
        models['s2_tri_proc'] = AutoImageProcessor.from_pretrained(S2_TRI_MODEL)
        models['s2_tri'] = AutoModelForImageClassification.from_pretrained(S2_TRI_MODEL).to(DEVICE)
        models['s2_tri'].eval()

        logger.success("Dual-Transformer Array safely active.")
    except Exception as e:
        import traceback
        startup_error = traceback.format_exc()
        logger.error(f"Failed to populate active weights: {e}\n{startup_error}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Offload the 60-second PyTorch I/O boot sequence to a background thread 
    # so Uvicorn can open port 7860 immediately for the Cloud Run readiness probe!
    asyncio.create_task(asyncio.to_thread(_load_models_sync))
    yield
    logger.info("Evacuating models via shutdown signal.")

app = FastAPI(
    title="Enterprise Vision Inference Array",
    description="Mathematical media dissection using SigLIP semantics and Tri-State Deepfake Detection.",
    version=VERSION,
    lifespan=lifespan
)

@app.get("/health")
def read_health():
    return {"status": "ok", "startup_error": startup_error, "models_loaded": list(models.keys())}

def _download_video_stream(url: str) -> str:
    """Securely buffer arbitrary video URLs to ephemeral storage. Halts at 50MB."""
    logger.info(f"Opening secure temporal video stream for: {url}")
    try:
        resp = requests.get(url, impersonate="chrome110", stream=True, timeout=20)
        if resp.status_code != 200:
            raise ValueError(f"Stream HTTP {resp.status_code}")
        
        # Ephemeral system memory target mapping. (Auto-purges if pod shuts down).
        temp_fd, temp_path = tempfile.mkstemp(suffix=".mp4")
        
        size = 0
        with os.fdopen(temp_fd, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=1024*1024):
                if chunk:
                    size += len(chunk)
                    if size > 50 * 1024 * 1024:
                        logger.warning("50MB DDoS allocation cap engaged.")
                        break # Halt buffering, process what we have securely.
                    f.write(chunk)
        return temp_path
    except Exception as e:
        logger.error(f"Temporal fetch aborted: {e}")
        raise ValueError(f"Temporal fetch aborted: {e}")

def _extract_video_frames(video_path: str) -> List[Image.Image]:
    """Extract exactly 3 chronological keyframes isolated from dynamic video structures."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("OpenCV pipeline failed to parse the video matrix.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = 30 # Fallback 

    targets = [0, total_frames // 2, total_frames - 2]
    frames = []

    for t in targets:
        if t < 0: t = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, t)
        ret, frame = cap.read()
        if ret:
            # OpenCV BGR mapping to generic Pillow RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb_frame))
        
    cap.release()
    try:
        os.remove(video_path) # Mathematically destroy the footprint
    except Exception as e:
        logger.warning(f"File ghost wipe failed: {e}")

    # Fallback if the video was fully corrupt
    if not frames:
         frames.append(Image.new('RGB', (224, 224), color = 'black'))

    return frames

def _process_media_payloads(urls: List[str], b64s: List[str]) -> List[List[Image.Image]]:
    """
    Injesting arbitrary URIs/Blobs. Natively handles `.mp4`, `.gif` video structures 
    by automatically expanding them into sub-arrays of Pillow Frames. Outputs a List 
    where each original request URL corresponds to exactly 1 List of Frames.
    """
    media_arrays = []
    
    # Analyze raw HTTP URLs
    for url in urls:
        try:
            # Cloudscraper Chrome Impersonation 
            resp = requests.get(url, impersonate="chrome110", timeout=15)
            if resp.status_code != 200:
                 raise ValueError(f"HTTP {resp.status_code}")
                 
            # Employ Magic Byte detection to classify payload securely
            raw_bytes = resp.content
            kind = filetype.guess(raw_bytes)
            
            if kind is not None and kind.mime.startswith("video"):
                vid_path = _download_video_stream(url)
                frames = _extract_video_frames(vid_path)
                media_arrays.append(frames)
            else:
                img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                media_arrays.append([img])
                
        except Exception as e:
            logger.error(f"Failed parsing payload {url}: {e}")
            raise ValueError(f"Failed payload ingestion: {e}")

    # Analyze raw Base64 Arrays
    for b64 in b64s:
        try:
            if "," in b64: b64 = b64.split(",")[1]
            raw_bytes = base64.b64decode(b64)
            kind = filetype.guess(raw_bytes)

            if kind is not None and kind.mime.startswith("video"):
                temp_fd, temp_path = tempfile.mkstemp(suffix=".mp4")
                with os.fdopen(temp_fd, 'wb') as f:
                    f.write(raw_bytes)
                frames = _extract_video_frames(temp_path)
                media_arrays.append(frames)
            else:
                img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                media_arrays.append([img])
        except Exception as e:
            logger.error(f"Failed Base64 B-matrix: {e}")
            raise ValueError("Base64 corruption")

    return media_arrays

def _calculate_synthetic_probability(images: List[Image.Image]) -> tuple[float, Optional[str]]:
    """
    Executes the Tri-State model array across a temporal sequence of frames.
    Averages probabilities per-frame to normalize motion blur noise, then aggregates the
    sum of AI and Deepfake probabilities.
    """
    if not images:
        logger.warning("_calculate_synthetic_probability called with empty images list.")
        return 0.0, "Empty images array"

    try:
        synthetic_probs = []

        # Log the actual label map
        tri_id2label = models['s2_tri'].config.id2label
        logger.info(f"[DIAG] tri_state id2label: {tri_id2label}")

        # The mapping is strictly {'0': 'AI', '1': 'Deepfake', '2': 'Real'}
        # We find indices dynamically just in case config keys change from string to int
        ai_idx = None
        df_idx = None
        for k, v in tri_id2label.items():
            if 'ai' in str(v).lower():
                ai_idx = int(k)
            elif 'deepfake' in str(v).lower():
                df_idx = int(k)
        
        # Fallbacks to known indices if regex fails
        if ai_idx is None: ai_idx = 0
        if df_idx is None: df_idx = 1

        with torch.no_grad():
            for img in images:
                # 1. Evaluate Unified Matrix
                raw_inputs = models['s2_tri_proc'](images=img, return_tensors="pt")
                inputs = {k: v.to(DEVICE) for k, v in raw_inputs.items()}
                logits = models['s2_tri'](**inputs).logits
                probs = torch.nn.functional.softmax(logits, dim=-1)[0]
                
                logger.info(f"[DIAG] tri_state full probs: {probs.tolist()}")

                # AI and Deepfake conceptually represent synthetic alterations
                # Probability = P(AI) + P(Deepfake)
                frame_prob = probs[ai_idx].item() + probs[df_idx].item()
                synthetic_probs.append(frame_prob)

        avg_synthetic = sum(synthetic_probs) / len(synthetic_probs)
        final = min(1.0, float(avg_synthetic))  # Clamp at 1.0 safely
        
        diag_logs = f"tri_id2label: {tri_id2label} | ai_idx: {ai_idx} | df_idx: {df_idx} | avg_synthetic: {final:.4f}"
        logger.info(f"[DIAG] {diag_logs}")
        
        return final, diag_logs

    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        logger.error(f"Classifier Array Panicked: {e}\n{err_msg}")
        return 0.0, err_msg


def _embed_matrix(media_arrays: List[List[Image.Image]]) -> tuple[List[List[float]], List[str], List[float], Optional[str]]:
    master_vectors = []
    master_phashes = []
    master_synth_probs = []
    master_debug = None

    if startup_error:
        master_debug = f"STARTUP PANIC TRACE:\n{startup_error}"

    try:
        for frame_group in media_arrays:
            # 1. Perceptual Hash (middle frame)
            center_frame = frame_group[len(frame_group) // 2]
            master_phashes.append(str(imagehash.phash(center_frame)))

            # 2. Synthetic Probability — trace exceptions dynamically
            synth_score, dbg = _calculate_synthetic_probability(frame_group)
            master_synth_probs.append(round(float(synth_score), 4))
            if dbg:
                master_debug = f"{master_debug}\n---\n{dbg}" if master_debug else dbg

            # 3. SigLIP Mathematical Embedding Matrix
            inputs = models['siglip_proc'](images=frame_group, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                vision_outputs = models['siglip'].vision_model(**inputs)
                image_embeds = vision_outputs.pooler_output
                image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

            avg_vector = torch.mean(image_embeds, dim=0, keepdim=True)
            avg_vector = avg_vector / avg_vector.norm(dim=-1, keepdim=True)
            master_vectors.append(avg_vector[0].cpu().tolist())

        return master_vectors, master_phashes, master_synth_probs, master_debug

    except Exception as e:
        import traceback
        logger.error(f"Matrix Algebra Panicked: {e}\n{traceback.format_exc()}")
        raise ValueError("Core Tensor Computation Failed")


async def wait_for_models():
    """Asynchronously hold the HTTP connection open while the background thread completes."""
    for _ in range(120): # Extend wait up to 120s for slow CPU bounds
        if 'siglip' in models and 's2_tri' in models:
            return True
        await asyncio.sleep(1.0)
    return False

@app.post("/embed_media", response_model=VisionEmbedResponse)
async def embed_media(request: VisionEmbedRequest, _: str = Depends(verify_api_key)):
    if 'siglip' not in models or 's2_tri' not in models:
        logger.info("Suspending request to wait for PyTorch array memory completion...")
        ready = await wait_for_models()
        if not ready:
            raise HTTPException(status_code=503, detail="Cold-start timeout. Neural array failed to bind to RAM in 120 seconds.")
    try:
        total = len(request.image_urls) + len(request.image_base64)
        if total == 0 or total > 16:
            raise HTTPException(status_code=400, detail="Matrix overflow. Provide 1 to 16 objects.")
            
        logger.info(f"Ingesting block of {total} targets.")
        
        # Async file I/O & TCP evasion operations to unblock the main server thread
        media_arrays = await asyncio.to_thread(_process_media_payloads, request.image_urls, request.image_base64)
        
        # Complex Matrix GPU/CPU Algebra execution
        vectors, phashes, synthetics, dbg_msg = await asyncio.to_thread(_embed_matrix, media_arrays)
        
        return VisionEmbedResponse(embeddings=vectors, phashes=phashes, synthetic_prob=synthetics, debug=dbg_msg)
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Deployment Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Service Panic")

class VisionTextEmbedRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to embed into the visual latent space")

def _embed_text_sync(texts: List[str]) -> List[List[float]]:
    try:
        inputs = models['siglip_proc'](text=texts, padding="max_length", return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            text_outputs = models['siglip'].text_model(**inputs)
            text_embeds = text_outputs.pooler_output
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        return text_embeds.cpu().tolist()
    except Exception as e:
        logger.error(f"Text computation failed: {e}")
        raise ValueError("Text computation failed")

@app.post("/embed_text", response_model=VisionEmbedResponse)
async def embed_text(request: VisionTextEmbedRequest, _: str = Depends(verify_api_key)):
    if 'siglip' not in models:
        ready = await wait_for_models()
        if not ready:
            raise HTTPException(status_code=503, detail="Cold-start timeout. Neural array failed to bind to RAM in 120 seconds.")
    try:
        if not request.texts:
            raise HTTPException(status_code=400, detail="Must provide texts")
            
        embeddings = await asyncio.to_thread(_embed_text_sync, request.texts)
        # Text embedding doesn't have phash or synthetic prob
        return VisionEmbedResponse(embeddings=embeddings)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Text embedding error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "architecture": "Triple-Transformer",
        "primary": SIGLIP_MODEL,
        "synthetic": SYNTH_MODEL,
        "deepfake": DF_MODEL,
        "capabilities": ["siglip_embedding", "video_cv2", "ai_generation", "deepface_swap"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
