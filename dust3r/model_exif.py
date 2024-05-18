# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# DUSt3R model class
# --------------------------------------------------------
from copy import deepcopy
import torch

from .utils.misc import fill_default_args, freeze_all_params, is_symmetrized, interleave, transpose_to_landscape
from .heads import head_factory
from dust3r.patch_embed import get_patch_embed

import dust3r.utils.path_to_croco  # noqa: F401
from models.croco import CroCoNet  # noqa
import exif_as_language.model_wrapper as exif
from croco.models.pos_embed import RoPE2D
from transformers import DistilBertTokenizer
inf = float('inf')


class AsymmetricCroCo3DStereoAndText (CroCoNet):
    """ Two siamese encoders, followed by two decoders.
    The goal is to output 3d points directly, both images in view1's frame
    (hence the asymmetry).
    """

    def __init__(self,
                 output_mode='pts3d',
                 head_type='linear',
                 depth_mode=('exp', -inf, inf),
                 conf_mode=('exp', 1, inf),
                 freeze='none',
                 landscape_only=True,
                 patch_embed_cls='PatchEmbedDust3R',  # PatchEmbedDust3R or ManyAR_PatchEmbed
                 exif_state_dict_path=None,
                 **croco_kwargs):
        self.patch_embed_cls = patch_embed_cls
        self.croco_args = fill_default_args(croco_kwargs, super().__init__)
        super().__init__(**croco_kwargs)

        # dust3r specific initialization
        self.dec_blocks2 = deepcopy(self.dec_blocks)
        self.set_downstream_head(output_mode, head_type, landscape_only, depth_mode, conf_mode, **croco_kwargs)
        self.set_freeze(freeze)

        self.text_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.text_encoder = exif.load_wrapper_model("cuda", state_dict_path=exif_state_dict_path)[0]
        self.text_position_encoder = self.text_encoder.model.transformer.get_position_embeddings()
        self.rope = RoPE2D()

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = get_patch_embed(self.patch_embed_cls, img_size, patch_size, enc_embed_dim)

    def load_state_dict(self, ckpt, **kw):
        # duplicate all weights for the second decoder if not present
        new_ckpt = dict(ckpt)
        if not any(k.startswith('dec_blocks2') for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith('dec_blocks'):
                    new_ckpt[key.replace('dec_blocks', 'dec_blocks2')] = value
        return super().load_state_dict(new_ckpt, **kw)

    def set_freeze(self, freeze):  # this is for use by downstream models
        self.freeze = freeze
        to_be_frozen = {
            'none':     [],
            'mask':     [self.mask_token],
            'encoder':  [self.mask_token, self.patch_embed, self.enc_blocks],
        }
        freeze_all_params(to_be_frozen[freeze])

    def _set_prediction_head(self, *args, **kwargs):
        """ No prediction head """
        return

    def set_downstream_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode, patch_size, img_size,
                            **kw):
        assert img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0, \
            f'{img_size=} must be multiple of {patch_size=}'
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        # allocate heads
        self.downstream_head1 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode), has_caption=True)
        self.downstream_head2 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode), has_caption=True)
        # magic wrapper
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)

    def _encode_image(self, image, true_shape):
        # embed the image into patches  (x has size B x Npatches x C)
        x, pos = self.patch_embed(image, true_shape=true_shape)

        # add positional embedding without cls token
        # assert self.enc_pos_embed is None

        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks:
            x = blk(x, pos)

        x = self.enc_norm(x)
        return x, pos, None

    def _encode_image_pairs(self, img1, img2, true_shape1, true_shape2):
        if img1.shape[-2:] == img2.shape[-2:]:
            out, pos, _ = self._encode_image(torch.cat((img1, img2), dim=0),
                                             torch.cat((true_shape1, true_shape2), dim=0))
            out, out2 = out.chunk(2, dim=0)
            pos, pos2 = pos.chunk(2, dim=0)
        else:
            out, pos, _ = self._encode_image(img1, true_shape1)
            out2, pos2, _ = self._encode_image(img2, true_shape2)
        return out, out2, pos, pos2

    def _encode_symmetrized(self, view1, view2):
        img1 = view1['img']
        img2 = view2['img']
        B = img1.shape[0]
        # Recover true_shape when available, otherwise assume that the img shape is the true one
        shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B, 1))
        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1))
        # warning! maybe the images have different portrait/landscape orientations

        if is_symmetrized(view1, view2):
            # computing half of forward pass!'
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1[::2], img2[::2], shape1[::2], shape2[::2])
            feat1, feat2 = interleave(feat1, feat2)
            pos1, pos2 = interleave(pos1, pos2)
        else:
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1, img2, shape1, shape2)

        return (shape1, shape2), (feat1, feat2), (pos1, pos2)

    def _decoder(self, f1, pos1, f2, pos2, encoded_caption1, position_caption1, encoded_caption2, position_caption2):
        final_output = [(f1, f2)]  # before projection

        # project to decoder dim
        f1 = self.decoder_embed(f1)
        encoded_caption1 = self.decoder_embed(encoded_caption1)
        f1 = torch.concatenate((f1, encoded_caption1[:, None, :]), dim=1)
        if len(position_caption1.size()) == 2:
            position_caption1 = position_caption1[None, :, :]
        if len(position_caption2.size()) == 2:
            position_caption2 = position_caption2[None, :, :]
        pos1 = torch.concatenate((pos1, position_caption1), dim=1)

        f2 = self.decoder_embed(f2)
        encoded_caption2 = self.decoder_embed(encoded_caption2)
        f2 = torch.concatenate((f2, encoded_caption2[:, None, :]), dim=1)
        pos2 = torch.concatenate((pos2, position_caption2), dim=1)

        final_output.append((f1, f2))
        for blk1, blk2 in zip(self.dec_blocks, self.dec_blocks2):
            # img1 side
            f1, _ = blk1(*final_output[-1][::+1], pos1, pos2)
            # img2 side
            f2, _ = blk2(*final_output[-1][::-1], pos2, pos1)
            # store the result
            final_output.append((f1, f2))

        # normalize last output
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))
        return zip(*final_output)

    def _downstream_head(self, head_num, decout, img_shape):
        B, S, D = decout[-1].shape
        # img_shape = tuple(map(int, img_shape))
        head = getattr(self, f'head{head_num}')
        return head(decout, img_shape)

    def forward(self, view1, view2):
        # encode the two images --> B,S,D
        (shape1, shape2), (feat1, feat2), (pos1, pos2) = self._encode_symmetrized(view1, view2)

        tokenized_caption1 = self.text_tokenizer(
            view1["caption"], truncation=True, padding="max_length", return_tensors="pt"
        ).to("cuda")
        tokenized_caption2 = self.text_tokenizer(
            view2["caption"], truncation=True, padding="max_length", return_tensors="pt"
        ).to("cuda")

        caption_1_pos = torch.zeros_like(tokenized_caption1["input_ids"]).to("cuda")
        for i in range(len(caption_1_pos)):
            caption_1_pos[i, :] = torch.arange(caption_1_pos.size()[1])

        caption_2_pos = torch.zeros_like(tokenized_caption2["input_ids"]).to("cuda")
        for i in range(len(caption_2_pos)):
            caption_2_pos[i, :] = torch.arange(caption_2_pos.size()[1])

        cos_cap1, sin_cap1 = self.rope.get_cos_sin(1, int(caption_1_pos.max()) + 1, tokenized_caption1["input_ids"].device, tokenized_caption1["input_ids"].dtype)
        cap1_rope = self.rope.apply_rope1d(tokenized_caption1["input_ids"][:, None, :, None], caption_1_pos, cos_cap1, sin_cap1).squeeze()

        cos_cap2, sin_cap2 = self.rope.get_cos_sin(1, int(caption_2_pos.max()) + 1, tokenized_caption2["input_ids"].device, tokenized_caption2["input_ids"].dtype)
        cap2_rope = self.rope.apply_rope1d(tokenized_caption2["input_ids"][:, None, :, None], caption_2_pos, cos_cap2, sin_cap2).squeeze()

        encoded_caption1 = self.text_encoder.encode_text({
            "attention_mask": tokenized_caption1["attention_mask"],
            "input_ids": tokenized_caption1["input_ids"]
        })

        positions_caption1 = self.text_position_encoder(torch.arange(encoded_caption1.size(0)).to(device="cuda"))

        encoded_caption2 = self.text_encoder.encode_text({
            "attention_mask": tokenized_caption2["attention_mask"],
            "input_ids": tokenized_caption2["input_ids"]
        })

        positions_caption2 = self.text_position_encoder(torch.arange(encoded_caption2.size(0)).to(device="cuda"))

        # combine all ref images into object-centric representation
        dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2, encoded_caption1, cap1_rope, encoded_caption2, cap2_rope)

        with torch.cuda.amp.autocast(enabled=False):
            res1 = self._downstream_head(1, [tok.float() for tok in dec1], shape1)
            res2 = self._downstream_head(2, [tok.float() for tok in dec2], shape2)

        res2['pts3d_in_other_view'] = res2.pop('pts3d')  # predict view2's pts3d in view1's frame
        return res1, res2
