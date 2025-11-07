# scripts/train_energy.py
from helper_lib.energy.trainer import train_energy

if __name__ == "__main__":
    train_energy(epochs=5)
    print("Energy model trained and checkpoints saved under checkpoints/energy/")


# export ENERGY_CKPT=checkpoints/energy/epoch_005.pt
# uv run uvicorn helper_lib.api:app --reload --port 8000


