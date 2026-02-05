import torch
from torch import nn
from transformers import AutoModel

class HybridBERT(nn.Module):
    def __init__(self, meta_input_dim=5, meta_hidden_dim=64):
        super(HybridBERT, self).__init__()
        self.bert = AutoModel.from_pretrained("bert-base-uncased")

        self.meta_fc = nn.Sequential(
            nn.Linear(meta_input_dim, meta_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.hidden = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size + meta_hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.classifier = nn.Linear(128, 1)

    def forward(self, input_ids, attention_mask, metadata):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = bert_output.last_hidden_state[:, 0, :]
        meta_out = self.meta_fc(metadata)
        combined = torch.cat((cls_output, meta_out), dim=1)
        hidden_out = self.hidden(combined)
        logits = self.classifier(hidden_out)
        return logits.squeeze()
