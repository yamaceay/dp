from transformers import LongformerModel
from torch import nn
import torch.nn as nn


class Model(nn.Module):
    """
    Full fine=tuning of all Longformer's parameters, with a linear classification layer on top.
    """
    def __init__(self, model, num_labels):
        super().__init__()
        self._bert = LongformerModel.from_pretrained(model)

        for param in self._bert.parameters():
           param.requires_grad = True

        self.classifier = nn.Linear(768, num_labels)
        
    def forward(self, batch):
        b = self._bert(
            input_ids=batch["input_ids"], attention_mask=batch["attention_masks"]
        )
        pooler = b.last_hidden_state
        return self.classifier(pooler)