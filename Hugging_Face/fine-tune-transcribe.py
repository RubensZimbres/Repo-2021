# Special thanks to gchhablani: https://colab.research.google.com/drive/1_BbLyLqDUsXG3RpSULfLRjC6UY3RjwME?usp=sharing#scrollTo=rQoRyVc9GGdY

import torch
import torchaudio
from datasets import load_dataset, load_metric
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import re

# test_dataset = #TODO: WRITE YOUR CODE TO LOAD THE TEST DATASET. For sample see the Colab link in Training Section.

wer = load_metric("wer")

processor = Wav2Vec2Processor.from_pretrained("gchhablani/wav2vec2-large-xlsr-mr")
model = Wav2Vec2ForCTC.from_pretrained("gchhablani/wav2vec2-large-xlsr-mr")
model.to("cuda")

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\–\…]'
resampler = torchaudio.transforms.Resample(48_000, 16_000)

# Preprocessing the datasets.
# We need to read the aduio files as arrays
def speech_file_to_array_fn(batch):
    batch["text"] = re.sub(chars_to_ignore_regex, '', batch["text"]).lower()
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    batch["speech"] = resampler(speech_array).squeeze().numpy()
    return batch

test_dataset = test_dataset.map(speech_file_to_array_fn)

# Preprocessing the datasets.
# We need to read the aduio files as arrays
def evaluate(batch):
    inputs = processor(batch["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values.to("cuda"), attention_mask=inputs.attention_mask.to("cuda")).logits
        pred_ids = torch.argmax(logits, dim=-1)
        batch["pred_strings"] = processor.batch_decode(pred_ids)
        return batch

result = test_dataset.map(evaluate, batched=True, batch_size=8)
# NOTE : The original test set was different and this result may be skewed. The original test set gave around 14.53% WER.
print("WER: {:2f}".format(100 * wer.compute(predictions=result["pred_strings"], references=result["text"])))
