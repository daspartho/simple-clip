from torch import nn
from transformers import (
    ViTConfig, 
    ViTModel, 
    ViTImageProcessor,
    BertConfig,
    BertModel,
    BertTokenizer,
)

class ImageEncoder(nn.Module):

    def __init__(self, image_size=224, patch_size=16, hidden_size=768, num_hidden_layers=12, num_attention_heads=12):
        super().__init__()

        self.image_processor = ViTImageProcessor(size={"height": image_size, "width": image_size})

        config = ViTConfig(
            image_size=image_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads
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
            num_attention_heads=num_attention_heads
        )
        self.model = BertModel(config)

    def forward(self, text):
        input = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True)
        output = self.model(input.input_ids)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, 0, :]