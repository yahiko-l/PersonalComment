from torch import nn
from transformers import RobertaConfig, RobertaModel, AutoModel

# from fengshen import LongformerModel
# from fengshen import LongformerConfig


class PAAEncoder(nn.Module):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        encoder_config = config.encoder
        
        if encoder_config.using_pretrained is not None:
            encoder = AutoModel.from_pretrained(encoder_config.using_pretrained)
        else:
            self.config_encoder = RobertaConfig()
            self.config_encoder.update(encoder_config)
            encoder = RobertaModel(config=self.config_encoder)
        encoder = self.resize_embedding(encoder)

        self.encoder = encoder

    def resize_embedding(self, transformer):
        transformer.resize_token_embeddings(len(self.tokenizer))
        return transformer

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state
        return last_hidden_states


class PretrainContentEncoder(nn.Module):
    def __init__(self, model_name_or_path):
        super().__init__()
        # self.tokenizer = tokenizer
        
        if "checkpoint" in model_name_or_path:
            config = RobertaConfig.from_pretrained(model_name_or_path)
            self.context_encoder = RobertaModel(config=config)
        else:
            self.context_encoder = RobertaModel.from_pretrained(model_name_or_path)

        # context_encoder = LongformerModel.from_pretrained("pretraining_model/Erlangshen-Longformer-110M")
        # self.context_encoder = self.resize_embedding(context_encoder)

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.context_encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state
        return last_hidden_states

    # def resize_embedding(self, transformer):
    #     transformer.resize_token_embeddings(len(self.tokenizer))
    #     return transformer


class PretrainPersonaEncoder(nn.Module):
    def __init__(self, model_name_or_path):
        super().__init__()
        if "checkpoint" in model_name_or_path:
            config = RobertaConfig.from_pretrained(model_name_or_path)
            self.persona_encoder = RobertaModel(config=config)
        else:
            self.persona_encoder = RobertaModel.from_pretrained(model_name_or_path)

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.persona_encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state
        return last_hidden_states
