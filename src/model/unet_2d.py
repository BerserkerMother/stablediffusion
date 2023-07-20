from diffusers import UNet2DModel

class U2Model:
    @classmethod
    def from_hf(cls, config):
        model = UNet2DModel(
            sample_size=config.image_size,  # the target image resolution
            in_channels=1,  # the number of input channels
            out_channels=1,  # the number of output channels
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            # the number of output channels for each UNet block
            block_out_channels=(64, 64, 128, 128, 256, 256),
            down_block_types=(
                "DownBlock2D",  # a regular ResNet downsampling block
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",  # a regular ResNet upsampling block
                "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )
        return model
