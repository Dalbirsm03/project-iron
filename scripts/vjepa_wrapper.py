import torch
import cProfile
import pstats
from transformers import AutoModel

class VJEPAWrapper:
    def __init__(self, model_path):
        print("Loading V-JEPA2 ViT-L model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    def forward(self, video_tensor):
        with torch.no_grad():
            output = self.model(video_tensor)
        return output


def profile_inference(wrapper, video_tensor):
    profiler = cProfile.Profile()
    profiler.enable()
    wrapper.forward(video_tensor)
    profiler.disable()

    stats = pstats.Stats(profiler)
    stats.sort_stats("cumtime").print_stats(20)


if __name__ == "__main__":

    model_dir = "../models/weights/vjepa2_vitl"

    wrapper = VJEPAWrapper(model_dir)

    # Dummy video tensor
    # Shape assumption: (batch, frames, channels, height, width)
    video_tensor = torch.randn(1, 8, 3, 224, 224).to(wrapper.device)

    print("Running profiling...")
    profile_inference(wrapper, video_tensor)
