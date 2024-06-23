python3 train.py --train_dataset MegaScenes('/share/phoenix/nfs06/S9/gc492/data/megascenes/pairs/info/', '/share/phoenix/nfs06/S9/am2283/', split='train', resolution=(224,224)) --test_dataset MegaScenes('/share/phoenix/nfs06/S9/gc492/data/megascenes/pairs/info/', '/share/phoenix/nfs06/S9/am2283/',split='test', resolution=(224, 224)) --model AsymmetricCroCo3DStereoAndText(patch_embed_cls='ManyAR_PatchEmbed', img_size=(224, 224), depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12) --output_dir checkpointed_out --duster_path checkpoints/dust3r/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth --exif_path checkpoints/exif/wrapper_75_new.pth --batch_size 8