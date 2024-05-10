import torch

from d3pm import D3PM

TEXT8_CHARS = list("_abcdefghijklmnopqrstuvwxyz█")

class Text8SampleDecoder:

    @staticmethod
    def char_ids_to_str(char_ids) -> str:
        """Decode a 1D sequence of character IDs to a string."""
        return "".join([TEXT8_CHARS[i] for i in char_ids.squeeze(-1)])

    @classmethod
    def batch_to_str(cls, text_batch) -> list[str]:
        """Decode a batch of character IDs to a list of strings."""
        return [cls.char_ids_to_str(row_char_ids) for row_char_ids in text_batch]


B, L = 1, 256
ckpt_path = ""  # full path to ckpt.pt

ckpt = torch.load(ckpt_path)
model_cls = dict(D3PM=D3PM)[ckpt["model_cls"]]
model_args = ckpt["model_args"]
model = model_cls(**model_args)
model.load_state_dict(ckpt["model"])
model.to("cuda").eval()

print("Sampling...")
trajectories = model.sample(B, L)

print("Sampled trajectory:")
for step_samples in trajectories:
    print(Text8SampleDecoder.batch_to_str(step_samples))
