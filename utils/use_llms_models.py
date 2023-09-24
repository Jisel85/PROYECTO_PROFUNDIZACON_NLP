import torch
from transformers import RobertaTokenizerFast, EncoderDecoderModel

device = "cuda" if torch.cuda.is_available() else "cpu"
model = "Narrativa/bsc_roberta2roberta_shared-spanish-finetuned-mlsum-summarization"
tokenizer = RobertaTokenizerFast.from_pretrained(model)
model = EncoderDecoderModel.from_pretrained(model).to(device)


def generate_summary_roberta(text):
    inputs = tokenizer(
        [text],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    output = model.generate(input_ids, attention_mask=attention_mask)
    return tokenizer.decode(output[0], skip_special_tokens=True)
