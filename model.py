import torch
from torch import nn
import torch.nn.functional as F
from transformers import (
    ViTConfig,
    ViTModel,
    ViTImageProcessor,
    BertConfig,
    BertModel,
    BertTokenizer,
)


class ImageEncoder(nn.Module):
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
    ):
        super().__init__()

        self.image_processor = ViTImageProcessor(
            size={"height": image_size, "width": image_size}
        )

        config = ViTConfig(
            image_size=image_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
        )
        self.model = ViTModel(config)

    def forward(self, image):
        input = self.image_processor(image, return_tensors="pt")
        output = self.model(input.pixel_values)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, 0, :]


class TextEncoder(nn.Module):
    def __init__(self, hidden_size=768, num_hidden_layers=12, num_attention_heads=12):
        super().__init__()

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        config = BertConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
        )
        self.model = BertModel(config)

    def forward(self, text):
        input = self.tokenizer(
            text, return_tensors="pt", padding="max_length", truncation=True
        )
        output = self.model(input.input_ids)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, 0, :]


class CLIPModel(nn.Module):
    def __init__(
        self,
        dim_latent=768,
        image_size=224,
        patch_size=16,
        image_encoder_hidden_size=768,
        image_encoder_num_hidden_layers=12,
        image_encoder_num_attention_heads=12,
        text_encoder_hidden_size=768,
        text_encoder_num_hidden_layers=12,
        text_encoder_num_attention_heads=12,
    ):
        super().__init__()

        self.image_encoder = ImageEncoder(
            image_size=image_size,
            patch_size=patch_size,
            hidden_size=image_encoder_hidden_size,
            num_hidden_layers=image_encoder_num_hidden_layers,
            num_attention_heads=image_encoder_num_attention_heads,
        )

        self.text_encoder = TextEncoder(
            hidden_size=text_encoder_hidden_size,
            num_hidden_layers=text_encoder_num_hidden_layers,
            num_attention_heads=text_encoder_num_attention_heads,
        )

        self.image_projection = nn.Linear(
            image_encoder_hidden_size, dim_latent, bias=False
        )
        self.text_projection = nn.Linear(
            text_encoder_hidden_size, dim_latent, bias=False
        )

        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, text, image):
        image_features = self.image_encoder(image)
        image_latents = F.normalize(self.image_projection(image_features))

        text_features = self.text_encoder(text)
        text_latents = F.normalize(self.text_projection(text_features))

        logits = (image_latents @ text_latents.T) * self.temperature.exp()
        labels = torch.arange(logits.size(dim=0))

        loss_image = F.cross_entropy(logits, labels)
        loss_text = F.cross_entropy(logits.T, labels)
        loss = (loss_image + loss_text) / 2
        return loss
