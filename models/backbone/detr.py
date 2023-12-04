from torch import Tensor, nn


class Transformer(nn.Transformer):
    def __init__(self, d_model: int) -> None:
        encoder_layer = TransformerEncoderLayer(d_model, nhead=8)
        custom_encoder = TransformerEncoder(encoder_layer, num_layers=6)
        decoder_layer = TransformerDecoderLayer(d_model, nhead=8)
        custom_decoder = TransformerDecoder(decoder_layer, num_layers=6, norm=nn.LayerNorm(d_model))
        super().__init__(d_model, custom_encoder=custom_encoder, custom_decoder=custom_decoder)

    def forward(self, src: Tensor, tgt: Tensor, *, pos_embed: Tensor, query_embed: Tensor) -> Tensor:
        memory = self.encoder(src, pos_embed=pos_embed)
        output = self.decoder(tgt, memory, pos_embed=pos_embed, query_embed=query_embed)
        return output


class TransformerEncoder(nn.TransformerEncoder):
    def forward(self, src: Tensor, pos_embed: Tensor) -> Tensor:
        output = src
        for mod in self.layers:
            output = mod(output, pos_embed=pos_embed)
        return output


class TransformerDecoder(nn.TransformerDecoder):
    def forward(self, tgt: Tensor, memory: Tensor, pos_embed: Tensor, query_embed: Tensor) -> Tensor:
        output = tgt
        for mod in self.layers:
            output = mod(output, memory, pos_embed=pos_embed, query_embed=query_embed)
        output = self.norm(output)
        return output


class TransformerEncoderLayer(nn.TransformerEncoderLayer):
    def forward(self, src: Tensor, pos_embed: Tensor) -> Tensor:
        x = src
        x = self.norm1(x + self._sa_block(x, pos_embed))
        x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(self, x: Tensor, pos_embed: Tensor) -> Tensor:
        q = k = x + pos_embed
        x = self.self_attn(q, k, x, need_weights=False)[0]
        return self.dropout1(x)


class TransformerDecoderLayer(nn.TransformerDecoderLayer):
    def forward(self, tgt: Tensor, memory: Tensor, pos_embed: Tensor, query_embed: Tensor) -> Tensor:
        x = tgt
        x = self.norm1(x + self._sa_block(x, query_embed))
        x = self.norm2(x + self._mha_block(x, memory, pos_embed, query_embed))
        x = self.norm3(x + self._ff_block(x))
        return x

    def _sa_block(self, x: Tensor, query_embed: Tensor) -> Tensor:
        q = k = x + query_embed
        x = self.self_attn(q, k, x, need_weights=False)[0]
        return self.dropout1(x)

    def _mha_block(self, x: Tensor, mem: Tensor, pos_embed: Tensor, query_embed: Tensor) -> Tensor:
        q = x + query_embed
        k = mem + pos_embed
        x = self.multihead_attn(q, k, mem, need_weights=False)[0]
        return self.dropout2(x)
