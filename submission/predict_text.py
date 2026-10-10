"""Use a saved local classifier on a new answer (optional demonstration).

Example:
python predict_text.py --run-dir results/full/runs/main_cnn_output_only_s42 --output-file answer.txt
For input_and_output also supply --prompt-file question.txt.
The five-way softmax does not detect a sixth/unknown model and is not calibrated.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from attribution.data import EncodedDataset, pad_batch
from attribution.engine import TextCNN, BiLSTM, choose_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-file", type=Path)
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    args = parser.parse_args()
    folder = args.run_dir.resolve()
    metadata = json.loads((folder / "run_manifest.json").read_text(encoding="utf-8"))
    cfg = metadata["config"]
    mode = metadata["mode"]
    if mode != "input_only" and args.output_file is None:
        parser.error("This classifier requires --output-file")
    if mode != "output_only" and args.prompt_file is None:
        parser.error("This classifier requires --prompt-file")
    prompt = args.prompt_file.read_text(encoding="utf-8-sig") if args.prompt_file else ""
    output = args.output_file.read_text(encoding="utf-8-sig") if args.output_file else ""
    if (mode != "input_only" and not output.strip()) or (mode != "output_only" and not prompt.strip()):
        parser.error("The required input text is empty")
    vocab = json.loads((folder / "vocabulary.json").read_text(encoding="utf-8"))
    labels = metadata["label_names"]
    kwargs = dict(vocab_size=max(vocab.values()) + 1,
                  embedding_dim=cfg["embedding_dim"], num_classes=len(labels), dropout=cfg["dropout"])
    if metadata["architecture"] == "cnn":
        model = TextCNN(**kwargs, filters=cfg["cnn_filters"], kernels=cfg["cnn_kernels"])
    else:
        model = BiLSTM(**kwargs, hidden_dim=cfg["lstm_hidden"])
    # The best-model file contains tensor weights and primitive metadata only.
    checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
    if checkpoint["signature"] != metadata["signature"]:
        raise ValueError("Checkpoint does not match this experiment directory")
    model.load_state_dict(checkpoint["model"])
    device = choose_device(args.device)
    model.to(device).eval()
    # label=0 is just an unused Dataset placeholder; no true label is supplied.
    frame = pd.DataFrame([{"LLM_Input": prompt, "LLM_output": output, "label": 0}])
    dataset = EncodedDataset(frame, vocab, mode, cfg,
                             transform=metadata["transform"], length_k=metadata.get("length_k"))
    inputs, lengths, _, _ = pad_batch([dataset[0]])
    with torch.inference_mode():
        probabilities = torch.softmax(model(inputs.to(device), lengths), dim=1)[0].cpu().tolist()
    order = sorted(range(len(labels)), key=lambda i: probabilities[i], reverse=True)
    print(f"Predicted source among these five classes: {labels[order[0]]}")
    for i in order:
        print(f"{labels[i]:10s} softmax score = {probabilities[i]:.4f}")
    print("These scores are not calibrated probabilities of authorship; unknown families are outside this classifier.")


if __name__ == "__main__":
    main()
