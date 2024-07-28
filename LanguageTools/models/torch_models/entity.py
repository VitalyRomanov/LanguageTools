from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Dict


try:
    import torch
    import torch.nn as nn
except ImportError:
    raise ImportError("Install torch: pip install torch")


from LanguageTools.models.torch_models.abstract_model import AbstractTorchModel


@dataclass
class ModelOutput:
    token_embs: torch.Tensor
    text_encoder_output: torch.Tensor
    logits: torch.Tensor
    prediction: torch.Tensor
    loss: Optional[torch.Tensor]
    scores: Optional[Dict]


# @dataclass
# class EntityAnnotatorConfigSpecification:
#     text_encoder_model: str = None
#     num_classes: int = None
#     ent_clf_hidden_dim: int = 100
#     ent_clf_dropout: float = 0.1
#     ent_clf_input_dim: int = 768


class EntityAnnotator(AbstractTorchModel):
    # config_specification = EntityAnnotatorConfigSpecification

    def __init__(
            self, *, text_encoder_model: str = None, num_classes: int = None, ent_clf_hidden_dim: int = 100,
            ent_clf_dropout: float = 0.1, ent_clf_input_dim: int = 768
    ):
        super(EntityAnnotator, self).__init__()
        self.config = {
            "text_encoder_model": text_encoder_model,
            "num_classes": num_classes,
            "ent_clf_hidden_dim": ent_clf_hidden_dim,
            "ent_clf_dropout": ent_clf_dropout,
            "ent_clf_input_dim": ent_clf_input_dim
        }

        self.encoder = text_encoder_model
        self.buffered_toke_type_ids = self.codebert_model.embeddings.token_type_ids.tile(1, 2)

        self.fc1 = nn.Linear(ent_clf_input_dim, ent_clf_hidden_dim)
        self.drop = nn.Dropout(ent_clf_dropout)
        self.fc2 = nn.Linear(ent_clf_hidden_dim, num_classes)

        self.loss_f = nn.CrossEntropyLoss(reduction="mean")

    @classmethod
    def get_default_config(cls):
        return deepcopy(cls.default_config)

    def forward(self, token_ids, graph_ids, mask, graph_mask, finetune=False, graph_embs=None):
        with torch.set_grad_enabled(finetune):
            token_embs_ = self.codebert_model.embeddings.word_embeddings(token_ids)
            position_ids = None
            if self.use_graph:
                graph_emb = self.graph_emb(graph_ids)
                position_ids = torch.arange(2, token_embs_.shape[1] + 2).reshape(1, -1).to(token_ids.device)
                graph_position_id = position_ids
                position_ids = torch.cat([position_ids, graph_position_id], dim=1)

                token_embs = torch.cat([token_embs_, self.graph_adapter(graph_emb)], dim=1)
                mask = torch.cat([mask, graph_mask], dim=1)
            else:
                token_embs = token_embs_
                graph_emb = None

            token_type_ids = self.buffered_toke_type_ids[:, :token_embs.size(1)].to(token_ids.device)
            codebert_output = self.codebert_model(
                inputs_embeds=token_embs,
                attention_mask=mask,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                output_attentions=True
            )

        x = codebert_output.last_hidden_state
        x = torch.relu(self.fc1(x[:, :token_ids.size(1), :]))
        x = self.drop(x)
        x = self.fc2(x)

        return ModelOutput(
            token_embs=token_embs_,
            text_encoder_output=codebert_output,
            logits=x,
            prediction=x.argmax(-1),
            loss=None, scores=None
        )

    def loss(self, logits, labels, mask, class_weights=None, extra_mask=None):
        if extra_mask is not None:
            mask = torch.logical_and(mask, extra_mask)
        logits = logits[mask, :]
        labels = labels[mask]
        loss = self.loss_f(logits, labels)
        # if class_weights is None:
        #     loss = tf.reduce_mean(tf.boolean_mask(losses, seq_mask))
        # else:
        #     loss = tf.reduce_mean(tf.boolean_mask(losses * class_weights, seq_mask))

        return loss

    def score(self, logits, labels, mask, scorer=None, extra_mask=None):
        if extra_mask is not None:
            mask = torch.logical_and(mask, extra_mask)
        true_labels = labels[mask]
        argmax = logits.argmax(-1)
        estimated_labels = argmax[mask]

        p, r, f1 = scorer(to_numpy(estimated_labels), to_numpy(true_labels))

        scores = {}
        scores["Precision"] = p
        scores["Recall"] = r
        scores["F1"] = f1

        return scores
