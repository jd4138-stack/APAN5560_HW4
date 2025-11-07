# scripts/train_diffusion.py
import argparse
from helper_lib.diffusion.trainer import train_diffusion

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints/diffusion")
    p.add_argument("--limit-train-batches", type=int, default=200)
    p.add_argument("--limit-val-batches", type=int, default=50)
    p.add_argument("--use-amp", action="store_true")
    p.add_argument("--no-amp", dest="use_amp", action="store_false")
    p.set_defaults(use_amp=True)
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()

def main():
    args = parse_args()
    train_diffusion(
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        use_amp=args.use_amp,
        num_workers=args.num_workers,
    )

if __name__ == "__main__":
    main()

#uv run python -m scripts.train_diffusion \
#  --image-size 32 \
#  --batch-size 32 \
#  --epochs 1 \
#  --limit-train-batches 100 \
#  --limit-val-batches 20 \
#  --use-amp

# export DIFFUSION_CKPT=checkpoints/diffusion/best.pt
# export DIFFUSION_IMAGE_SIZE=32
# export DIFFUSION_CHANNELS=3
# export DIFFUSION_MODEL=base
# uv run uvicorn helper_lib.api:app --reload --port 8000