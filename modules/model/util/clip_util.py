from torch import Tensor

from transformers import CLIPTextModel, CLIPTextModelWithProjection


def encode_clip(
        text_encoder: CLIPTextModel | CLIPTextModelWithProjection,
        tokens: Tensor | None = None,
        default_layer: int = -1,
        layer_skip: int = 0,
        add_output: bool = True,
        text_encoder_output: Tensor | None = None,
        add_pooled_output: bool = False,
        pooled_text_encoder_output: Tensor | None = None,
        use_attention_mask: bool = True,
        attention_mask: Tensor | None = None,
        add_layer_norm: bool = True,
) -> tuple[Tensor, Tensor]:
    if (add_output and text_encoder_output is None) \
            or (add_pooled_output and pooled_text_encoder_output is None) \
            and text_encoder is not None:

        batch_size, num_chunks = tokens.shape[0], 1
        if tokens.dim() > 2:
            num_chunks = tokens.shape[1]
            tokens = tokens.reshape(batch_size * num_chunks, -1)

        text_encoder_output = text_encoder(
            tokens,
            attention_mask=attention_mask if use_attention_mask else None,
            return_dict=True,
            output_hidden_states=True,
        )

        pooled_text_encoder_output = None
        if add_pooled_output:
            if hasattr(text_encoder_output, "text_embeds"):
                pooled_text_encoder_output = text_encoder_output.text_embeds
            if hasattr(text_encoder_output, "pooler_output"):
                pooled_text_encoder_output = text_encoder_output.pooler_output

            if pooled_text_encoder_output is not None and num_chunks > 1:
                # Extract pooler output of first chunk from each batch
                pooled_text_encoder_output = pooled_text_encoder_output[::num_chunks, :].clone()

        text_encoder_output = text_encoder_output.hidden_states[default_layer - layer_skip] if add_output else None

        if add_layer_norm and text_encoder_output is not None:
            final_layer_norm = text_encoder.text_model.final_layer_norm
            text_encoder_output = final_layer_norm(text_encoder_output)

        if num_chunks > 1:
            num_features = text_encoder_output.shape[-1]
            text_encoder_output = text_encoder_output.reshape(batch_size, -1, num_features)

    return text_encoder_output, pooled_text_encoder_output
