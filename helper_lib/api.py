from fastapi import FastAPI, HTTPException, Response, UploadFile, File, Query
import os, base64
from io import BytesIO
import torch
from torchvision import transforms
from torchvision.utils import make_grid
from helper_lib.shared.checkpoints import load_checkpoint
from helper_lib.gan.model import DCGANGenerator
from helper_lib.energy.model import EnergyModel
from helper_lib.energy.model import generate_samples
# -----faster diffusion model-----
import os
from helper_lib.diffusion.model import UNet, DiffusionModel, cosine_diffusion_schedule
try:
    from helper_lib.diffusion.model import UNetTiny  # optional tiny backbone for fast runs
except Exception:
    UNetTiny = None

from helper_lib.shared.checkpoints import load_checkpoint

# --- env-configurable diffusion config (defaults match original 64x64 RGB) ---
DIFFUSION_IMAGE_SIZE = int(os.getenv("DIFFUSION_IMAGE_SIZE", "64"))   # e.g. 32 for faster model
DIFFUSION_CHANNELS   = int(os.getenv("DIFFUSION_CHANNELS", "3"))      # 1 for grayscale, 3 for RGB
DIFFUSION_MODEL_KIND = os.getenv("DIFFUSION_MODEL", "base").lower()   # "base" or "tiny"

# optional classifier 
try:
    from helper_lib.classifier.model import get_model as get_classifier
    HAVE_CLS = True
except Exception:
    HAVE_CLS = False

app = FastAPI(title="SPS GenAI: Unified API")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STATE = {
    "gan":        {"G": None, "ckpt": os.getenv("GAN_G_CKPT", "checkpoints/gan/epoch_020.pt")},
    "classifier": {"M": None, "ckpt": os.getenv("CLASSIFIER_CKPT")},
    "energy":     {"M": None, "ckpt": os.getenv("ENERGY_CKPT")},
    "diffusion":  {"M": None, "ckpt": os.getenv("DIFFUSION_CKPT")},
}

@app.get("/health")
def health():
    return {k: {"loaded": (STATE[k]["G"] is not None) if k=="gan" else (STATE[k]["M"] is not None),
                "ckpt": STATE[k]["ckpt"]} for k in STATE}

def _ensure_gan(z_dim=100):
    if STATE["gan"]["G"] is None:
        ckpt = STATE["gan"]["ckpt"]
        if not ckpt or not os.path.exists(ckpt): raise HTTPException(503, "GAN not loaded; set GAN_G_CKPT")
        G = DCGANGenerator(z_dim); _ = load_checkpoint(G, optimizer=None, checkpoint_path=ckpt, device=DEVICE)
        G.to(DEVICE).eval(); STATE["gan"]["G"] = G
    return STATE["gan"]["G"]

def _ensure_classifier(num_classes=10):
    if not HAVE_CLS: raise HTTPException(503, "Classifier code not present")
    if STATE["classifier"]["M"] is None:
        ckpt = STATE["classifier"]["ckpt"]
        if not ckpt or not os.path.exists(ckpt): raise HTTPException(503, "Classifier not loaded; set CLASSIFIER_CKPT")
        M = get_classifier("CNN", num_classes); _ = load_checkpoint(M, optimizer=None, checkpoint_path=ckpt, device=DEVICE)
        M.to(DEVICE).eval(); STATE["classifier"]["M"] = M
    return STATE["classifier"]["M"]

def _ensure_energy():
    if STATE["energy"]["M"] is None:
        ckpt = STATE["energy"]["ckpt"]
        if not ckpt or not os.path.exists(ckpt):
            raise HTTPException(503, "Energy model not loaded; set ENERGY_CKPT")
        M = EnergyModel()
        _ = load_checkpoint(M, optimizer=None, checkpoint_path=ckpt, device=DEVICE)
        M.to(DEVICE).eval()
        STATE["energy"]["M"] = M
    return STATE["energy"]["M"]

def _ensure_diffusion():
    """
    Lazy-load the diffusion model using env-configured architecture and checkpoint.
    Env vars:
      DIFFUSION_CKPT         -> path to .pt/.pth (required)
      DIFFUSION_IMAGE_SIZE   -> e.g. 32 or 64 (default 64)
      DIFFUSION_CHANNELS     -> e.g. 1 or 3  (default 3)
      DIFFUSION_MODEL        -> "base" or "tiny" (default "base")
    """
    if STATE["diffusion"]["M"] is not None:
        return STATE["diffusion"]["M"]

    ckpt = STATE["diffusion"]["ckpt"]
    if not ckpt or not os.path.exists(ckpt):
        raise HTTPException(503, f"Diffusion model not loaded; set DIFFUSION_CKPT (got: {ckpt!r})")

    # Build backbone consistent with how you trained it
    if DIFFUSION_MODEL_KIND == "tiny":
        if UNetTiny is None:
            raise HTTPException(500, "UNetTiny not available; import failed. Use DIFFUSION_MODEL=base or add UNetTiny.")
        net = UNetTiny(image_size=DIFFUSION_IMAGE_SIZE, num_channels=DIFFUSION_CHANNELS)
    else:
        net = UNet(image_size=DIFFUSION_IMAGE_SIZE, num_channels=DIFFUSION_CHANNELS)

    M = DiffusionModel(net, schedule_fn=cosine_diffusion_schedule).to(DEVICE)
    try:
        # Load wrapper state (your shared loader handles model/optimizer dicts)
        _ = load_checkpoint(M, optimizer=None, checkpoint_path=ckpt, device=DEVICE)
    except Exception as e:
        # Fallback: try loading a plain state_dict if your checkpoint was saved that way
        try:
            state = torch.load(ckpt, map_location=DEVICE)
            sd = state.get("model_state_dict") or state.get("state_dict") or state
            M.load_state_dict(sd)
        except Exception as ee:
            raise HTTPException(500, f"Failed to load diffusion checkpoint: {e} / {ee}")

    # Normalizer defaults (match training; adjust if you trained differently)
    M.set_normalizer(mean=0.5, std=0.5)
    M.eval()

    STATE["diffusion"]["M"] = M
    return M


@app.get("/v1/infer/gan")
def infer_gan(n: int = 16, nrow: int = 4, z_dim: int = 100, return_base64: bool = True):
    G = _ensure_gan(z_dim)
    with torch.no_grad():
        z = torch.randn(n, z_dim, device=DEVICE)
        imgs = G(z).cpu()
        grid = make_grid(imgs, nrow=nrow, normalize=True, value_range=(-1, 1))
        pil = transforms.ToPILImage()(grid)
        buf = BytesIO(); pil.save(buf, "PNG")
    if return_base64:
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return {"image_base64": b64, "n": n, "nrow": nrow}
    return Response(buf.getvalue(), media_type="image/png")

@app.post("/v1/infer/classifier")
async def infer_classifier(file: UploadFile = File(...)):
    M = _ensure_classifier(10)
    from PIL import Image
    import torch.nn.functional as F
    img = Image.open(BytesIO(await file.read())).convert("RGB")
    tf = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize([0.4914,0.4822,0.4465],[0.2470,0.2435,0.2616]),
    ])
    x = tf(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        p = F.softmax(M(x), dim=1)[0]
        top_p, top_i = p.topk(3)
    labels = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
    return {"topk":[{"label": labels[i.item()], "index": int(i), "prob": float(pp)} for pp,i in zip(top_p, top_i)]}


@app.get("/v1/infer/energy/sample", tags=["Energy"], summary="Generate samples via EBM Langevin dynamics")
def energy_sample(n: int = 16, nrow: int = 4, steps: int = 40, step_size: float = 10.0,
                  noise: float = 0.005, return_base64: bool = True):
    M = _ensure_energy()  # loads EnergyModel() from ENERGY_CKPT
    device = DEVICE

    # start from uniform noise in [-1, 1]
    with torch.no_grad():
        x0 = torch.rand(n, 1, 28, 28, device=device) * 2 - 1

    # Langevin dynamics using ∇_x E(x)
    xk = generate_samples(M, x0, steps=steps, step_size=step_size, noise_std=noise)

    # return a grid as PNG (base64 or raw)
    grid = make_grid(xk.cpu(), nrow=nrow, normalize=True, value_range=(-1, 1))
    pil = transforms.ToPILImage()(grid)
    buf = BytesIO(); pil.save(buf, "PNG")
    if return_base64:
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return {"image_base64": b64, "n": n, "nrow": nrow, "steps": steps, "step_size": step_size, "noise": noise}
    return Response(buf.getvalue(), media_type="image/png")


@app.get("/v1/infer/diffusion", tags=["Diffusion"], summary="Generate samples (diffusion)")
def infer_diffusion(n: int = 16, nrow: int = 4, steps: int = 50, return_base64: bool = True):
    M = _ensure_diffusion()
    with torch.no_grad():
        imgs = M.generate(num_images=n, diffusion_steps=steps, image_size=DIFFUSION_IMAGE_SIZE)
        grid = make_grid(imgs, nrow=nrow)  # normalize not needed if [0,1]
        pil = transforms.ToPILImage()(grid)
        buf = BytesIO(); pil.save(buf, "PNG")
    if return_base64:
        import base64
        return {"image_base64": base64.b64encode(buf.getvalue()).decode("ascii"),
                "n": n, "nrow": nrow, "steps": steps}
    return Response(buf.getvalue(), media_type="image/png")



