import torch
from sentence_transformers import SentenceTransformer
from pathlib import Path

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
ONNX_PATH = Path("index/model.onnx")

print("Loading SentenceTransformer...")
st_model = SentenceTransformer(MODEL_NAME)
tokenizer = st_model.tokenizer
bert_model = st_model[0].auto_model

class EmbedWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        # out[0] is last_hidden_state: [batch, seq_len, hidden_dim]
        # Mean pooling:
        token_embeddings = out[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask
        # Normalize:
        return torch.nn.functional.normalize(mean_pooled, p=2, dim=1)

wrapper = EmbedWrapper(bert_model).eval()

dummy_inputs = tokenizer("test query for export", return_tensors="pt")

print("Exporting complete model with mean-pooling to ONNX...")
torch.onnx.export(
    wrapper,
    (dummy_inputs["input_ids"], dummy_inputs["attention_mask"]),
    str(ONNX_PATH),
    input_names=["input_ids", "attention_mask"],
    output_names=["sentence_embedding"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "sentence_embedding": {0: "batch_size"},
    },
    opset_version=14,
    do_constant_folding=True,
    dynamo=False,
)

print(f"Exported successfully to {ONNX_PATH}")

