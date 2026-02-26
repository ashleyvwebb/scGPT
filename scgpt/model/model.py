import gc
import math
from typing import Dict, Mapping, Optional, Tuple, Any, Union

import torch
import numpy as np
from torch import nn, Tensor
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.distributions import Bernoulli
from tqdm import trange

try:
    from flash_attn.flash_attention import FlashMHA

    flash_attn_available = True
except ImportError:
    import warnings

    warnings.warn("flash_attn is not installed")
    flash_attn_available = False

from .dsbn import DomainSpecificBatchNorm1d
from .grad_reverse import grad_reverse

# TODO: look into this class 
class TransformerModel(nn.Module):
    def __init__(
        self,
        ntoken: int, # vocab size for gene tokens
        d_model: int, # embedding dimension - or number of expected features in the encoder/decoder inputs
        nhead: int, # number of attention heads
        d_hid: int, # the dimensions of the feedfowrard network model
        nlayers: int, # number of stacked transformer encoder/decoder layers
        nlayers_cls: int = 3, # number of layers in the classification decoder
        n_cls: int = 1, # size of output used in classification decoder
        vocab: Any = None, # vocabulary object used
        dropout: float = 0.5, # the dropout value 
        pad_token: str = "<pad>", # used to fill shorter sequences in a bit, to ensure all cell input vectors have the same uniform length
        pad_value: int = 0, # pad value for expression values
        do_mvc: bool = False, # determines whether the model should do masked value prediction for cell embedding - also known as GEPC
        do_dab: bool = False, # determines whether to use a domain adversarial loss to ensure that the learned cell embeddings are independent of the batch ID
        use_batch_labels: bool = False, # whether to incorporate batch-specific info
        num_batch_labels: Optional[int] = None, # number of unique batch labels, required if use_batch_labels is True
        domain_spec_batchnorm: Union[bool, str] = False, # domain specific batch normalization or not, can be True/False or "dsbn"/"do_affine"/"batchnorm"
        input_emb_style: str = "continuous", # how to encode the input expression values, can be "continuous", "category" or "scaling"
        n_input_bins: Optional[int] = None, # number of bins to discretize the input expression values if input_emb_style is "category"
        cell_emb_style: str = "cls", # how to get the cell embedding from the transformer output, can be "cls", "avg-pool" or "w-pool"
        mvc_decoder_style: str = "inner product", # the architecture style for the masked value prediction decoder, can be "inner product" or "mlp"
        ecs_threshold: float = 0.3, # Elastic Cell Similarity threshold, similarity between cells in a latent space
        explicit_zero_prob: bool = False, # whether to explicitly model the zero values with a separate probability output, which is used in both MLM and MVC decoders. If True, the model will have an additional output "zero_probs" for both MLM and MVC, which can be used to sample zero values during generation.
        use_fast_transformer: bool = False,
        fast_transformer_backend: str = "flash",
        pre_norm: bool = False, # determines position of layer normalisation within each transformer block, when true, layer normalisation is applied before the self-attention and feed-forward sub-layers. Otherwise it is applied after 
    ):
        super().__init__()
        self.model_type = "Transformer"
        self.d_model = d_model
        self.do_dab = do_dab
        self.ecs_threshold = ecs_threshold
        self.use_batch_labels = use_batch_labels
        self.domain_spec_batchnorm = domain_spec_batchnorm
        self.input_emb_style = input_emb_style
        self.cell_emb_style = cell_emb_style
        self.explicit_zero_prob = explicit_zero_prob
        self.norm_scheme = "pre" if pre_norm else "post"
        if self.input_emb_style not in ["category", "continuous", "scaling"]:
            raise ValueError(
                f"input_emb_style should be one of category, continuous, scaling, "
                f"got {input_emb_style}"
            )
        if cell_emb_style not in ["cls", "avg-pool", "w-pool"]:
            raise ValueError(f"Unknown cell_emb_style: {cell_emb_style}")
        if use_fast_transformer:
            if not flash_attn_available:
                warnings.warn(
                    "flash-attn is not installed, using pytorch transformer instead. "
                    "Set use_fast_transformer=False to avoid this warning. "
                    "Installing flash-attn is highly recommended."
                )
                use_fast_transformer = False
        self.use_fast_transformer = use_fast_transformer

        # TODO: add dropout in the GeneEncoder
        self.encoder = GeneEncoder(ntoken, d_model, padding_idx=vocab[pad_token])


        # Value Encoder, NOTE: the scaling style is also handled in _encode method
        if input_emb_style == "continuous": # the whole human pretrained model uses this style
            self.value_encoder = ContinuousValueEncoder(d_model, dropout)
        elif input_emb_style == "category": # ? not as important to look at for the time being
            assert n_input_bins > 0
            self.value_encoder = CategoryValueEncoder(
                n_input_bins, d_model, padding_idx=pad_value
            )
        else:
            self.value_encoder = nn.Identity()  # nn.Softmax(dim=1)
            # TODO: consider row-wise normalization or softmax
            # TODO: Correct handle the mask_value when using scaling

        # Batch Encoder
        if use_batch_labels: # ! assume this is false in whole human model as cannot see it in the arguments list
            self.batch_encoder = BatchLabelEncoder(num_batch_labels, d_model)

        if domain_spec_batchnorm is True or domain_spec_batchnorm == "dsbn": # ! assume this is False in whole human model as cannot see it in the arguments list
            use_affine = True if domain_spec_batchnorm == "do_affine" else False
            print(f"Use domain specific batchnorm with affine={use_affine}")
            self.dsbn = DomainSpecificBatchNorm1d(
                d_model, num_batch_labels, eps=6.1e-5, affine=use_affine
            )
        elif domain_spec_batchnorm == "batchnorm":
            print("Using simple batchnorm instead of domain specific batchnorm")
            self.bn = nn.BatchNorm1d(d_model, eps=6.1e-5)

        if use_fast_transformer: # this was true in the training - but in theory shouldnt have any difference between it being true or not
            if fast_transformer_backend == "linear": # backend not defined so would default to flash
                self.transformer_encoder = FastTransformerEncoderWrapper(
                    d_model, nhead, d_hid, nlayers, dropout
                )
            elif fast_transformer_backend == "flash":
                encoder_layers = FlashTransformerEncoderLayer(
                    d_model,
                    nhead,
                    d_hid,
                    dropout,
                    batch_first=True,
                    norm_scheme=self.norm_scheme, # pre-norm for the whole human pretrained model
                )
                self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers) # stack nlayers of the same encoder layer, which is the standard transformer architecture. The encoder layer is defined above as FlashTransformerEncoderLayer, which is modified from the standard TransformerEncoderLayer to support flash attention and flexible normalization scheme.
        else:
            encoder_layers = TransformerEncoderLayer(
                d_model, nhead, d_hid, dropout, batch_first=True
            )
            self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.decoder = ExprDecoder(
            d_model,
            explicit_zero_prob=explicit_zero_prob,
            use_batch_labels=use_batch_labels,
        ) # the decoder for the masked language modeling (MLM) objective, which predicts the gene expression values based on the transformer output. The decoder will have a linear layer to predict the expression values, and if explicit_zero_prob is True, it will also have a separate output for the probability of zero values. If use_batch_labels is True, the batch embedding will be concatenated to the transformer output before feeding into the decoder.
        self.cls_decoder = ClsDecoder(d_model, n_cls, nlayers=nlayers_cls)
        if do_mvc: # determines whether the model should do masked value prediction for cell embedding - also known as GEPC. If True, the model will have an additional decoder (MVCDecoder) to predict the expression values of the masked genes based on the cell embedding. This is a key component for the GEPC objective, which aims to improve the quality of the learned cell embeddings by predicting the expression values of masked genes.
            self.mvc_decoder = MVCDecoder(
                d_model,
                arch_style=mvc_decoder_style,
                explicit_zero_prob=explicit_zero_prob,
                use_batch_labels=use_batch_labels,
            )

        if do_dab:
            self.grad_reverse_discriminator = AdversarialDiscriminator(
                d_model,
                n_cls=num_batch_labels,
                reverse_grad=True,
            ) # the discriminator used for the domain adversarial loss, which aims to make the learned cell embeddings independent of the batch ID. The discriminator takes the cell embedding as input and tries to predict the batch label. The grad_reverse flag indicates whether to reverse the gradients during backpropagation, which is a common technique in domain adversarial training to encourage the encoder to learn batch-invariant features.

        self.sim = Similarity(temp=0.5)  # the similarity function used for the contrastive cell embedding objective (CCE), which computes the cosine similarity between cell embeddings. The temperature parameter temp is used to scale the similarities before applying the contrastive loss, which can help improve the training of the contrastive objective.
        self.creterion_cce = nn.CrossEntropyLoss() # the loss function used for the contrastive cell embedding objective (CCE), which is a cross-entropy loss applied to the similarity scores between cell embeddings. The CCE objective encourages the model to learn cell embeddings that are similar for cells of the same type and dissimilar for cells of different types, which can improve the quality of the learned cell representations.

        self.init_weights() # initialise the weights of the model for the gene encoder to be between 0.1 and -0.1 

    def init_weights(self) -> None:
        initrange = 0.1
        # TODO: check if this initialization is helpful and shall we apply to all?
        self.encoder.embedding.weight.data.uniform_(-initrange, initrange)
        # this fills the weight tensor with numbers smapled from the continuous unifrom distribution f(x) = 1 /(-0.1 - 0.1)

    def _encode(
        self,
        src: Tensor,
        values: Tensor,
        src_key_padding_mask: Tensor,
        batch_labels: Optional[Tensor] = None,  # (batch,)
    ) -> Tensor:
        self._check_batch_labels(batch_labels)

        src = self.encoder(src)  # (batch, seq_len, embsize)
        self.cur_gene_token_embs = src

        values = self.value_encoder(values)  # (batch, seq_len, embsize)
        if self.input_emb_style == "scaling":
            values = values.unsqueeze(2)
            total_embs = src * values
        else:
            total_embs = src + values

        if getattr(self, "dsbn", None) is not None:
            batch_label = int(batch_labels[0].item())
            total_embs = self.dsbn(total_embs.permute(0, 2, 1), batch_label).permute(
                0, 2, 1
            )  # the batch norm always works on dim 1
        elif getattr(self, "bn", None) is not None:
            total_embs = self.bn(total_embs.permute(0, 2, 1)).permute(0, 2, 1)

        output = self.transformer_encoder(
            total_embs, src_key_padding_mask=src_key_padding_mask
        )
        return output  # (batch, seq_len, embsize)

    def _get_cell_emb_from_layer(
        self, layer_output: Tensor, weights: Tensor = None
    ) -> Tensor:
        """
        Args:
            layer_output(:obj:`Tensor`): shape (batch, seq_len, embsize)
            weights(:obj:`Tensor`): shape (batch, seq_len), optional and only used
                when :attr:`self.cell_emb_style` is "w-pool".

        Returns:
            :obj:`Tensor`: shape (batch, embsize)
        """
        if self.cell_emb_style == "cls":
            cell_emb = layer_output[:, 0, :]  # (batch, embsize)
        elif self.cell_emb_style == "avg-pool":
            cell_emb = torch.mean(layer_output, dim=1)
        elif self.cell_emb_style == "w-pool":
            if weights is None:
                raise ValueError("weights is required when cell_emb_style is w-pool")
            if weights.dim() != 2:
                raise ValueError("weights should be 2D")
            cell_emb = torch.sum(layer_output * weights.unsqueeze(2), dim=1)
            cell_emb = F.normalize(cell_emb, p=2, dim=1)  # (batch, embsize)

        return cell_emb

    def _check_batch_labels(self, batch_labels: Tensor) -> None:
        if self.use_batch_labels or self.domain_spec_batchnorm:
            assert batch_labels is not None
        elif batch_labels is not None:
            raise ValueError(
                "batch_labels should only be provided when `self.use_batch_labels`"
                " or `self.domain_spec_batchnorm` is True"
            )

    def generate(
        self,
        cell_emb: Tensor,
        src: Tensor,
        values: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        gen_iters: int = 1,
        batch_labels: Optional[Tensor] = None,  # (batch,)
    ) -> Tensor:
        """
        Args:
            cell_emb(:obj:`Tensor`): shape (batch, embsize)
            src(:obj:`Tensor`): shape (batch, seq_len)
            values(:obj:`Tensor`): shape (batch, seq_len), optional
            src_key_padding_mask(:obj:`Tensor`): shape (batch, seq_len), optional
            gen_iters(:obj:`int`): number of generation iterations
            batch_labels(:obj:`Tensor`): shape (batch,), optional
        """
        # TODO: should have a tag indicate the generation mode
        # TODO: if gen_iters > 1, should have a tag indicate the current iteration
        try:
            self._check_batch_labels(batch_labels)
        except:
            import warnings

            warnings.warn(
                "batch_labels is required but not provided, using zeros instead"
            )
            batch_labels = torch.zeros(
                cell_emb.shape[0], dtype=torch.long, device=cell_emb.device
            )

        src = self.encoder(src)  # (batch, seq_len, embsize)

        if values is not None:
            values = self.value_encoder(values)  # (batch, seq_len, embsize)
            if self.input_emb_style == "scaling":
                values = values.unsqueeze(2)
                total_embs = src * values
            else:
                total_embs = src + values
        else:
            total_embs = src

        if getattr(self, "dsbn", None) is not None:
            batch_label = int(batch_labels[0].item())
            total_embs = self.dsbn(total_embs.permute(0, 2, 1), batch_label).permute(
                0, 2, 1
            )  # the batch norm always works on dim 1
        elif getattr(self, "bn", None) is not None:
            total_embs = self.bn(total_embs.permute(0, 2, 1)).permute(0, 2, 1)

        total_embs[:, 0, :] = cell_emb

        if src_key_padding_mask is None:
            src_key_padding_mask = torch.zeros(
                total_embs.shape[:2], dtype=torch.bool, device=total_embs.device
            )
        transformer_output = self.transformer_encoder(
            total_embs, src_key_padding_mask=src_key_padding_mask
        )

        if self.use_batch_labels:
            batch_emb = self.batch_encoder(batch_labels)  # (batch, embsize)
        mlm_output = self.decoder(
            transformer_output
            if not self.use_batch_labels
            else torch.cat(
                [
                    transformer_output,
                    batch_emb.unsqueeze(1).repeat(1, transformer_output.shape[1], 1),
                ],
                dim=2,
            ),
            # else transformer_output + batch_emb.unsqueeze(1),
        )
        output = mlm_output["pred"]  # (batch, seq_len)

        return output  # (batch, seq_len)

    def forward(
        self,
        src: Tensor,
        values: Tensor,
        src_key_padding_mask: Tensor,
        batch_labels: Optional[Tensor] = None,
        CLS: bool = False,
        CCE: bool = False,
        MVC: bool = False,
        ECS: bool = False,
        do_sample: bool = False,
    ) -> Mapping[str, Tensor]:
        """
        Args:
            src (:obj:`Tensor`): token ids, shape [batch_size, seq_len]
            values (:obj:`Tensor`): token values, shape [batch_size, seq_len]
            src_key_padding_mask (:obj:`Tensor`): mask for src, shape [batch_size,
                seq_len]
            batch_labels (:obj:`Tensor`): batch labels, shape [batch_size]
            CLS (:obj:`bool`): if True, return the celltype classification objective
                (CLS) output
            CCE (:obj:`bool`): if True, return the contrastive cell embedding objective
                (CCE) output
            MVC (:obj:`bool`): if True, return the masked value prediction for cell
                embedding MVC output
            ECS (:obj:`bool`): if True, return the elastic cell similarity objective
                (ECS) output.

        Returns:
            dict of output Tensors.
        """
        transformer_output = self._encode(
            src, values, src_key_padding_mask, batch_labels
        )
        if self.use_batch_labels:
            batch_emb = self.batch_encoder(batch_labels)  # (batch, embsize)

        output = {}
        mlm_output = self.decoder(
            transformer_output
            if not self.use_batch_labels
            else torch.cat(
                [
                    transformer_output,
                    batch_emb.unsqueeze(1).repeat(1, transformer_output.shape[1], 1),
                ],
                dim=2,
            ),
            # else transformer_output + batch_emb.unsqueeze(1),
        )
        if self.explicit_zero_prob and do_sample:
            bernoulli = Bernoulli(probs=mlm_output["zero_probs"])
            output["mlm_output"] = bernoulli.sample() * mlm_output["pred"]
        else:
            output["mlm_output"] = mlm_output["pred"]  # (batch, seq_len)
        if self.explicit_zero_prob:
            output["mlm_zero_probs"] = mlm_output["zero_probs"]

        cell_emb = self._get_cell_emb_from_layer(transformer_output, values)
        output["cell_emb"] = cell_emb

        if CLS:
            output["cls_output"] = self.cls_decoder(cell_emb)  # (batch, n_cls)
        if CCE:
            cell1 = cell_emb
            transformer_output2 = self._encode(
                src, values, src_key_padding_mask, batch_labels
            )
            cell2 = self._get_cell_emb_from_layer(transformer_output2)

            # Gather embeddings from all devices if distributed training
            if dist.is_initialized() and self.training:
                cls1_list = [
                    torch.zeros_like(cell1) for _ in range(dist.get_world_size())
                ]
                cls2_list = [
                    torch.zeros_like(cell2) for _ in range(dist.get_world_size())
                ]
                dist.all_gather(tensor_list=cls1_list, tensor=cell1.contiguous())
                dist.all_gather(tensor_list=cls2_list, tensor=cell2.contiguous())

                # NOTE: all_gather results have no gradients, so replace the item
                # of the current rank with the original tensor to keep gradients.
                # See https://github.com/princeton-nlp/SimCSE/blob/main/simcse/models.py#L186
                cls1_list[dist.get_rank()] = cell1
                cls2_list[dist.get_rank()] = cell2

                cell1 = torch.cat(cls1_list, dim=0)
                cell2 = torch.cat(cls2_list, dim=0)
            # TODO: should detach the second run cls2? Can have a try
            cos_sim = self.sim(cell1.unsqueeze(1), cell2.unsqueeze(0))  # (batch, batch)
            labels = torch.arange(cos_sim.size(0)).long().to(cell1.device)
            output["loss_cce"] = self.creterion_cce(cos_sim, labels)
        if MVC:
            mvc_output = self.mvc_decoder(
                cell_emb
                if not self.use_batch_labels
                else torch.cat([cell_emb, batch_emb], dim=1),
                # else cell_emb + batch_emb,
                self.cur_gene_token_embs,
            )
            if self.explicit_zero_prob and do_sample:
                bernoulli = Bernoulli(probs=mvc_output["zero_probs"])
                output["mvc_output"] = bernoulli.sample() * mvc_output["pred"]
            else:
                output["mvc_output"] = mvc_output["pred"]  # (batch, seq_len)
            if self.explicit_zero_prob:
                output["mvc_zero_probs"] = mvc_output["zero_probs"]
        if ECS:
            # Here using customized cosine similarity instead of F.cosine_similarity
            # to avoid the pytorch issue of similarity larger than 1.0, pytorch # 78064
            # normalize the embedding
            cell_emb_normed = F.normalize(cell_emb, p=2, dim=1)
            cos_sim = torch.mm(cell_emb_normed, cell_emb_normed.t())  # (batch, batch)

            # mask out diagnal elements
            mask = torch.eye(cos_sim.size(0)).bool().to(cos_sim.device)
            cos_sim = cos_sim.masked_fill(mask, 0.0)
            # only optimize positive similarities
            cos_sim = F.relu(cos_sim)

            output["loss_ecs"] = torch.mean(1 - (cos_sim - self.ecs_threshold) ** 2)

        if self.do_dab:
            output["dab_output"] = self.grad_reverse_discriminator(cell_emb)

        return output

    def encode_batch(
        self,
        src: Tensor,
        values: Tensor,
        src_key_padding_mask: Tensor,
        batch_size: int,
        batch_labels: Optional[Tensor] = None,
        output_to_cpu: bool = True,
        time_step: Optional[int] = None,
        return_np: bool = False,
    ) -> Tensor:
        """
        Args:
            src (Tensor): shape [N, seq_len]
            values (Tensor): shape [N, seq_len]
            src_key_padding_mask (Tensor): shape [N, seq_len]
            batch_size (int): batch size for encoding
            batch_labels (Tensor): shape [N, n_batch_labels]
            output_to_cpu (bool): whether to move the output to cpu
            time_step (int): the time step index in the transformer output to return.
                The time step is along the second dimenstion. If None, return all.
            return_np (bool): whether to return numpy array

        Returns:
            output Tensor of shape [N, seq_len, embsize]
        """
        N = src.size(0)
        device = next(self.parameters()).device

        # initialize the output tensor
        array_func = np.zeros if return_np else torch.zeros
        float32_ = np.float32 if return_np else torch.float32
        shape = (
            (N, self.d_model)
            if time_step is not None
            else (N, src.size(1), self.d_model)
        )
        outputs = array_func(shape, dtype=float32_)

        for i in trange(0, N, batch_size):
            raw_output = self._encode(
                src[i : i + batch_size].to(device),
                values[i : i + batch_size].to(device),
                src_key_padding_mask[i : i + batch_size].to(device),
                batch_labels[i : i + batch_size].to(device)
                if batch_labels is not None
                else None,
            )
            output = raw_output.detach()
            if output_to_cpu:
                output = output.cpu()
            if return_np:
                output = output.numpy()
            if time_step is not None:
                output = output[:, time_step, :]
            outputs[i : i + batch_size] = output

        return outputs


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)


class FastTransformerEncoderWrapper(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.fast_transformer_encoder = self.build_fast_transformer_encoder(
            d_model, nhead, d_hid, nlayers, dropout
        )

    @staticmethod
    def build_fast_transformer_encoder(
        d_model: int, nhead: int, d_hid: int, nlayers: int, dropout: float
    ) -> nn.Module:
        from fast_transformers.builders import TransformerEncoderBuilder

        if d_model % nhead != 0:
            raise ValueError(
                f"d_model must be divisible by nhead, "
                f"got d_model={d_model} and nhead={nhead}"
            )
        builder = TransformerEncoderBuilder.from_kwargs(
            n_layers=nlayers,
            n_heads=nhead,
            query_dimensions=d_model // nhead,
            value_dimensions=d_model // nhead,
            feed_forward_dimensions=d_hid,
            attention_type="linear",
            attention_dropout=dropout,
            dropout=dropout,
            activation="gelu",
        )
        assert builder.attention_type == "linear"
        return builder.get()

    @staticmethod
    def build_length_mask(
        src: Tensor,
        src_key_padding_mask: torch.BoolTensor,
    ) -> "LengthMask":
        from fast_transformers.masking import LengthMask

        seq_len = src.shape[1]
        num_paddings = src_key_padding_mask.sum(dim=1)
        actual_seq_len = seq_len - num_paddings  # (N,)
        length_mask = LengthMask(actual_seq_len, max_len=seq_len, device=src.device)

        if src_key_padding_mask[length_mask.bool_matrix].sum() != 0:
            raise ValueError(
                "Found padding tokens in the middle of the sequence. "
                "src_key_padding_mask and length_mask are not compatible."
            )
        return length_mask

    def forward(
        self,
        src: Tensor,
        src_key_padding_mask: torch.BoolTensor,
    ) -> Tensor:
        """
        Args:
            src: Tensor, shape [N, seq_len, embsize]
            src_key_padding_mask: Tensor, shape [N, seq_len]

        Returns:
            output Tensor of shape [N, seq_len, embsize]
        """
        if src_key_padding_mask.shape != src.shape[:2]:
            raise ValueError(
                f"src_key_padding_mask shape {src_key_padding_mask.shape} "
                f"does not match first two dims of src shape {src.shape[:2]}"
            )

        if src_key_padding_mask.dtype != torch.bool:
            raise ValueError(
                f"src_key_padding_mask needs to be of type torch.bool, "
                f"got {src_key_padding_mask.dtype}"
            )

        length_mask = self.build_length_mask(src, src_key_padding_mask)
        output = self.fast_transformer_encoder(src, length_mask=length_mask)
        return output


class FlashTransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    The class is modified from torch.nn.TransformerEncoderLayer to support the
    FlashAttention.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``.

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    """
    __constants__ = ["batch_first"]

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        layer_norm_eps=1e-5,
        batch_first=True,
        device=None,
        dtype=None,
        norm_scheme="post",  # "pre" or "post"
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.self_attn = FlashMHA(
            embed_dim=d_model,
            num_heads=nhead,
            batch_first=batch_first,
            attention_dropout=dropout,
            **factory_kwargs,
        )
        # Version compatibility workaround
        if not hasattr(self.self_attn, "batch_first"):
            self.self_attn.batch_first = batch_first
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = self._get_activation_fn(activation)
        self.norm_scheme = norm_scheme
        if self.norm_scheme not in ["pre", "post"]:
            raise ValueError(f"norm_scheme should be pre or post, not {norm_scheme}")

    @staticmethod
    def _get_activation_fn(activation):
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu

        raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super().__setstate__(state)

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        if src_mask is not None:
            raise ValueError("FlashTransformerEncoderLayer does not support src_mask")

        if not src_key_padding_mask.any().item():
            # no padding tokens in src
            src_key_padding_mask_ = None
        else:
            if src_key_padding_mask.dtype != torch.bool:
                src_key_padding_mask = src_key_padding_mask.bool()
            # NOTE: the FlashMHA uses mask 0 for padding tokens, which is the opposite
            src_key_padding_mask_ = ~src_key_padding_mask

        if self.norm_scheme == "pre":
            src = self.norm1(src)
            src2 = self.self_attn(src, key_padding_mask=src_key_padding_mask_)[0]
            src = src + self.dropout1(src2)
            src = self.norm2(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
        else:
            src2 = self.self_attn(src, key_padding_mask=src_key_padding_mask_)[0]
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)

        return src


class GeneEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int, # number of genes in the vocab
        embedding_dim: int,  # number of expected features in the input
        padding_idx: Optional[int] = None, # the index of the padding token in the vocab
    ):
        super().__init__() # initialise internal module state of the parent class
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        ) # A simple lookup table that stores embeddings of a fixed dictionary and size. Use to store the gene token embeddings and retrieve them based on the input gene token ids. The embedding layer will be randomly initialized and then trained together with the rest of the model. The embedding for the padding token will be initialized to zeros and will not be updated during training.
        self.enc_norm = nn.LayerNorm(embedding_dim) # Applies Layer Normalisation over a mini-batch of inputs. it takes an input the shape of the input the layer should expect.

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # (batch, seq_len, embsize)
        x = self.enc_norm(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class ContinuousValueEncoder(nn.Module):
    """
    Encode real number values to a vector using neural nets projection.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_value: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout) # during training, randomly zeroes some of the elements of the input tensor with probability p
        self.linear1 = nn.Linear(1, d_model) # applies an affine linear transformation to the incoming data: y = xA^T + b. Here the input is a single real value (hence the input dimension is 1) and the output is a vector of dimension d_model. This is used to encode the continuous values associated with each gene token into a vector that can be added to the gene token embedding.
        self.activation = nn.ReLU() # Applies the ReLU function element-wise. ReLU(x) = max(0, x). This is used as the activation function after the first linear layer to introduce non-linearity into the value encoding.
        self.linear2 = nn.Linear(d_model, d_model) # another linear layer that takes the output of the first linear layer (after activation) and projects it back to the same dimension d_model. This allows for more complex transformations of the input value.
        self.norm = nn.LayerNorm(d_model) # applies Layer Normalisation over a mini-batch of inputs. This is used to stabilize the training of the value encoder by normalizing the output of the second linear layer.
        self.max_value = max_value

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len]
        """
        # TODO: test using actual embedding layer if input is categorical
        # expand last dimension
        x = x.unsqueeze(-1) # change the shape from [batch_size, seq_len] to [batch_size, seq_len, 1] so that it can be fed into the linear layer which expects input of shape [batch_size, seq_len, input_dim]. The input_dim here is 1 because we are encoding a single real value for each gene token.
        # clip x to [-inf, max_value]
        x = torch.clamp(x, max=self.max_value) # clip the input values to a maximum value to prevent extremely large values that could destabilize training. The minimum value is not set, so it can be as low as -inf.
        x = self.activation(self.linear1(x)) # apply the first linear layer to project the input value to a d_model dimensional vector, then apply the ReLU activation function to introduce non-linearity.
        x = self.linear2(x) # apply the second linear layer to further transform the value encoding. The output is still of dimension d_model.
        x = self.norm(x) # apply layer normalization to the output of the second linear layer to stabilize training. This normalizes the output across the feature dimension (d_model) for each token in the sequence.
        return self.dropout(x) # apply dropout to the final output of the value encoder. This helps prevent overfitting by randomly zeroing out some of the elements in the output vector during training.


class CategoryValueEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x.long()
        x = self.embedding(x)  # (batch, seq_len, embsize)
        x = self.enc_norm(x)
        return x


class BatchLabelEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # (batch, embsize)
        x = self.enc_norm(x)
        return x


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp # this does the dot product of the two vectors along the last dimension and then divides by the temperature. The cosine similarity is computed as the dot product of the two vectors divided by the product of their magnitudes. By dividing by the temperature, we can control the scale of the similarity scores, which can be important for training stability and performance in contrastive learning settings.

#! Very Important for the masking strategy - we could run this forward on a single cell's data in order to get the predicted expression values for the genes - confrim this with the forward pass for the model and the training process
class ExprDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        explicit_zero_prob: bool = False,
        use_batch_labels: bool = False,
    ):
        super().__init__()
        d_in = d_model * 2 if use_batch_labels else d_model # if use_batch_labels is True, the input to the decoder will be the concatenation of the cell embedding and the batch embedding, so the input dimension will be 2 * d_model. Otherwise, the input dimension will be d_model.
        self.fc = nn.Sequential(
            nn.Linear(d_in, d_model), # hidden layer 1
            nn.LeakyReLU(),
            nn.Linear(d_model, d_model), # hidden layer 2
            nn.LeakyReLU(),
            nn.Linear(d_model, 1), # output layer
        ) # a simple feedforward neural network with two hidden layers and LeakyReLU activation functions. The output layer has a single unit which will predict the expression value for each gene token. The input dimension is d_in, which depends on whether batch labels are used or not.
        self.explicit_zero_prob = explicit_zero_prob
        if explicit_zero_prob:
            self.zero_logit = nn.Sequential( # logit is the raw, unnormlaised output of a model before applying a sigmoid or softmax.
                nn.Linear(d_in, d_model),
                nn.LeakyReLU(),
                nn.Linear(d_model, d_model),
                nn.LeakyReLU(),
                nn.Linear(d_model, 1),
            ) # if explicit_zero_prob is True, an additional feedforward network is defined to predict the logit for the probability of the expression being zero. This network has the same architecture as the main prediction network but outputs a single logit which will be passed through a sigmoid function to get the probability of zero expression.

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """x is the output of the transformer, (batch, seq_len, d_model)"""
        pred_value = self.fc(x).squeeze(-1)  # (batch, seq_len) the predicted expression values for each gene token. The output of the fc network is of shape (batch, seq_len, 1) and we squeeze the last dimension to get (batch, seq_len).

        if not self.explicit_zero_prob:
            return dict(pred=pred_value) # if explicit_zero_prob is False, we simply return the predicted expression values in a dictionary with the key "pred". The zero probability is not explicitly modeled in this case.
        zero_logits = self.zero_logit(x).squeeze(-1)  # (batch, seq_len) the predicted logits for the probability of zero expression for each gene token. Similar to pred_value, we squeeze the last dimension to get (batch, seq_len).
        zero_probs = torch.sigmoid(zero_logits) # apply the sigmoid function to the zero logits to get the probabilities of zero expression. The sigmoid function maps the logits to a range between 0 and 1, which can be interpreted as probabilities.
        return dict(pred=pred_value, zero_probs=zero_probs) # we return both the predicted expression values and the zero probabilities in a dictionary. The key "pred" corresponds to the predicted expression values, and the key "zero_probs" corresponds to the probabilities of zero expression for each gene token. This allows the model to explicitly account for the possibility of zero expression, which can be important in single-cell gene expression data where many genes may not be expressed in a given cell.
        # TODO: note that the return currently is only for training. Since decoder
        # is not used in the test setting for the integration task, the eval/inference
        # logic is not implemented yet. However, remember to implement it when
        # the decoder is used in any test setting. The inference logic will need
        # to sample from the bernoulli distribution with the zero_probs.


class ClsDecoder(nn.Module): # the classification tasks are the cell type classification and the batch classification.
    """
    Decoder for classification task.
    """

    def __init__(
        self,
        d_model: int,
        n_cls: int,
        nlayers: int = 3,
        activation: callable = nn.ReLU,
    ):
        super().__init__() # initialise internal module state of the parent class
        # module list
        self._decoder = nn.ModuleList() # a list to hold the layers of the decoder. We will append linear layers, activation functions, and layer normalization layers to this list in a loop based on the number of layers specified by nlayers.
        for i in range(nlayers - 1): # use a ModuleList as the number of layers is determined by the nlayers parameter. We loop through nlayers - 1 because the last layer will be the output layer which is defined separately after the loop.
            self._decoder.append(nn.Linear(d_model, d_model))
            self._decoder.append(activation())
            self._decoder.append(nn.LayerNorm(d_model))
        self.out_layer = nn.Linear(d_model, n_cls) # the output layer of the decoder which maps the final hidden representation to the number of classes (n_cls). This will be used to predict the class probabilities for each cell based on its embedding.

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, embsize]
        """
        for layer in self._decoder:
            x = layer(x) # run each layer in the decoder sequentially, this cannot be done using sequential as we dont know how many sets of these layers we will need.
        return self.out_layer(x) # after passing through the hidden layers, we apply the output layer to get the final class logits. The output shape will be [batch_size, n_cls], which can then be used to compute the classification loss during training.


class MVCDecoder(nn.Module):
    """
    Decoder for the masked value prediction for cell embeddings.
    """

    def __init__(
        self,
        d_model: int,
        arch_style: str = "inner product",
        query_activation: nn.Module = nn.Sigmoid,
        hidden_activation: nn.Module = nn.PReLU,
        explicit_zero_prob: bool = False,
        use_batch_labels: bool = False,
    ) -> None:
        """
        Args:
            d_model (:obj:`int`): dimension of the gene embedding.
            arch_style (:obj:`str`): architecture style of the decoder, choice from
                1. "inner product" or 2. "concat query" or 3. "sum query".
            query_activation (:obj:`nn.Module`): activation function for the query
                vectors.
            hidden_activation (:obj:`nn.Module`): activation function for the hidden
                layers.
        """
        super().__init__()
        d_in = d_model * 2 if use_batch_labels else d_model
        if arch_style in ["inner product", "inner product, detach"]:
            self.gene2query = nn.Linear(d_model, d_model) # a linear layer that projects the gene embeddings to query vectors. The input and output dimensions are both d_model, which means we are transforming the gene embeddings into a new space of the same dimension to be used as queries for the inner product calculation.
            self.query_activation = query_activation() # the activation function applied to the query vectors after the linear transformation. This can be a non-linear activation function like Sigmoid or ReLU, which can help the model learn more complex relationships between the gene embeddings and the cell embedding.
            self.W = nn.Linear(d_model, d_in, bias=False) # a linear layer that projects the query vectors to the same dimension as the cell embedding (d_in). This layer is used to compute the inner product between the query vectors and the cell embedding to predict the expression values. The bias is set to False because we want the inner product to be purely based on the transformed query vectors and the cell embedding without any additional bias term.
            if explicit_zero_prob:  # by default, gene-wise prob rate
                self.W_zero_logit = nn.Linear(d_model, d_in)
        elif arch_style == "concat query":
            self.gene2query = nn.Linear(d_model, 64)
            self.query_activation = query_activation()
            self.fc1 = nn.Linear(d_model + 64, 64)
            self.hidden_activation = hidden_activation()
            self.fc2 = nn.Linear(64, 1)
        elif arch_style == "sum query":
            self.gene2query = nn.Linear(d_model, d_model)
            self.query_activation = query_activation()
            self.fc1 = nn.Linear(d_model, 64)
            self.hidden_activation = hidden_activation()
            self.fc2 = nn.Linear(64, 1)
        else:
            raise ValueError(f"Unknown arch_style: {arch_style}")

        self.arch_style = arch_style
        self.do_detach = arch_style.endswith("detach")
        self.explicit_zero_prob = explicit_zero_prob

    def forward(
        self, cell_emb: Tensor, gene_embs: Tensor
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """
        Args:
            cell_emb: Tensor, shape (batch, embsize=d_model)
            gene_embs: Tensor, shape (batch, seq_len, embsize=d_model)
        """
        gene_embs = gene_embs.detach() if self.do_detach else gene_embs
        if self.arch_style in ["inner product", "inner product, detach"]:
            query_vecs = self.query_activation(self.gene2query(gene_embs)) # (batch, seq_len, embsize) the query vectors obtained by applying a linear transformation and an activation function to the gene embeddings. These query vectors will be used to compute the inner product with the cell embedding to predict the expression values for each gene token.
            cell_emb = cell_emb.unsqueeze(2)  # (batch, embsize, 1)
            # the pred gene expr values, # (batch, seq_len)
            # ? torch.bmm is batch matrix multiplication, it performs a matrix multiplication for each sample in the batch. The first argument is of shape (batch, seq_len, embsize) and the second argument is of shape (batch, embsize, 1), so the output will be of shape (batch, seq_len, 1). We then squeeze the last dimension to get (batch, seq_len) which represents the predicted expression values for each gene token in the sequence for each cell in the batch.
            pred_value = torch.bmm(self.W(query_vecs), cell_emb).squeeze(2)
            if not self.explicit_zero_prob:
                return dict(pred=pred_value)
            # zero logits need to based on the cell_emb, because of input exprs
            zero_logits = torch.bmm(self.W_zero_logit(query_vecs), cell_emb).squeeze(2)
            zero_probs = torch.sigmoid(zero_logits)
            return dict(pred=pred_value, zero_probs=zero_probs)
        elif self.arch_style == "concat query":
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            # expand cell_emb to (batch, seq_len, embsize)
            cell_emb = cell_emb.unsqueeze(1).expand(-1, gene_embs.shape[1], -1)

            h = self.hidden_activation(
                self.fc1(torch.cat([cell_emb, query_vecs], dim=2))
            )
            if self.explicit_zero_prob:
                raise NotImplementedError
            return self.fc2(h).squeeze(2)  # (batch, seq_len)
        elif self.arch_style == "sum query":
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            cell_emb = cell_emb.unsqueeze(1)

            h = self.hidden_activation(self.fc1(cell_emb + query_vecs))
            if self.explicit_zero_prob:
                raise NotImplementedError
            return self.fc2(h).squeeze(2)  # (batch, seq_len)


class AdversarialDiscriminator(nn.Module):
    """
    Discriminator for the adversarial training for batch correction.
    """

    def __init__(
        self,
        d_model: int,
        n_cls: int,
        nlayers: int = 3,
        activation: callable = nn.LeakyReLU,
        reverse_grad: bool = False,
    ):
        super().__init__()
        # module list
        self._decoder = nn.ModuleList()
        for i in range(nlayers - 1): # use a ModuleList as the number of layers is determined by the nlayers parameter. We loop through nlayers - 1 because the last layer will be the output layer which is defined separately after the loop.
            self._decoder.append(nn.Linear(d_model, d_model))
            self._decoder.append(activation())
            self._decoder.append(nn.LayerNorm(d_model))
        self.out_layer = nn.Linear(d_model, n_cls) # the output layer of the discriminator which maps the final hidden representation to the number of classes (n_cls). This will be used to predict the batch labels for each cell based on its embedding. During adversarial training, we want to minimize the classification loss for the main task (e.g., gene expression prediction) while maximizing the classification loss for the batch prediction, which encourages the model to learn batch-invariant features in the cell embeddings.
        self.reverse_grad = reverse_grad

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, embsize]
        """
        if self.reverse_grad:
            x = grad_reverse(x, lambd=1.0) # apply gradient reversal to the input embeddings if reverse_grad is True. This is used during adversarial training to reverse the gradients for the batch prediction task, which encourages the model to learn features that are not predictive of the batch labels.
        for layer in self._decoder:
            x = layer(x) # run each layer in the discriminator sequentially, this cannot be done using sequential as we dont know how many sets of these layers we will need.
        return self.out_layer(x) # after passing through the hidden layers, we apply the output layer to get the final class logits for the batch prediction. The output shape will be [batch_size, n_cls], which can then be used to compute the adversarial loss during training.
