import torch
import torch.nn as nn
import math

# ############################################################################
# This file defines the neural network architectures for our project.
# It includes:
# 1. UNet: The primary model for the inversion task (Seismic -> Velocity Map).
# 2. ForwardModel: A model to simulate the forward problem (Velocity Map -> Seismic),
#    used for the cycle consistency loss.
# 3. TransformerGenerator: A Vision Transformer (ViT) based model for a
#    fundamentally different approach to the problem.
# ############################################################################


def conv_block(in_channels, out_channels):
    """A helper function for a standard double-convolution block."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """
    A U-Net-like architecture for the primary INVERSE problem.

    Task: Transform seismic data into a velocity map.
    Input Shape:  (Batch, 5, 1000, 70)  -> (Channels, Time, Geophones)
    Output Shape: (Batch, 1, 70, 70)    -> (Channels, Height, Width)

    The architecture uses an encoder-decoder structure with skip connections
    to ensure both high-level context and fine-grained spatial details
    are captured. The main challenge is handling the unconventional input
    shape (1000x70) and mapping it to a square output (70x70).
    """

    def __init__(self, in_channels=5, out_channels=1):
        super(UNet, self).__init__()

        # --- Encoder (Downsampling Path) ---
        self.enc1 = conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)) # Reduces time: 1000->500

        self.enc2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Reduces both: 500x70 -> 250x35

        self.enc3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # 250x35 -> 125x17 (approx)

        # --- Bottleneck ---
        self.bottleneck = conv_block(256, 512)

        # --- Decoder (Upsampling Path) ---
        # Note: The Transposed Convolutions must be carefully designed to
        # reconstruct the spatial dimensions back to 70x70. This skeleton
        # shows the structure, but kernel sizes/strides may need tuning.
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = conv_block(512, 256)  # 256 from upconv + 256 from enc3

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = conv_block(256, 128)  # 128 from upconv + 128 from enc2

        # --- Final Output Layer ---
        # This layer maps the final feature map to the desired output channels.
        self.final_conv = nn.Conv2d(128, out_channels, kernel_size=1)

    def forward(self, x):
        # This forward pass is a schematic. In a real implementation, care must be
        # taken to handle size mismatches at skip connections, likely by padding
        # or cropping the encoder feature map.

        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        # Bottleneck
        b = self.bottleneck(p3)

        # Decoder
        d3 = self.upconv3(b)
        # Resize e3 to match the spatial dimensions of d3 before concatenating
        e3_resized = nn.functional.interpolate(e3, size=d3.shape[2:], mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, e3_resized], dim=1)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        # Resize e2 to match the spatial dimensions of d2 before concatenating
        e2_resized = nn.functional.interpolate(e2, size=d2.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, e2_resized], dim=1)
        d2 = self.dec2(d2)

        # In a complete model, we would continue upsampling until the spatial
        # dimensions are close to 70x70, then use interpolation or a final
        # transposed convolution to get the exact size.
        output = self.final_conv(d2)

        # For this skeleton, we assume `output` needs to be resized.
        # A common way to force the final size is with interpolation.
        return nn.functional.interpolate(output, size=(70, 70), mode='bilinear', align_corners=False)


class UNetV2(nn.Module):
    """
    A second, deeper version of the U-Net.
    This version has double the feature channels in each block, giving it
    more capacity to learn complex patterns.
    """

    def __init__(self, in_channels=5, out_channels=1):
        super(UNetV2, self).__init__()

        # --- Encoder (Downsampling Path) ---
        self.enc1 = conv_block(in_channels, 128)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        self.enc2 = conv_block(128, 256)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = conv_block(256, 512)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Bottleneck ---
        self.bottleneck = conv_block(512, 1024)

        # --- Decoder (Upsampling Path) ---
        self.upconv3 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec3 = conv_block(1024, 512)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2 = conv_block(512, 256)

        # --- Final Output Layer ---
        self.final_conv = nn.Conv2d(256, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        # Bottleneck
        b = self.bottleneck(p3)

        # Decoder
        d3 = self.upconv3(b)
        e3_resized = nn.functional.interpolate(e3, size=d3.shape[2:], mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, e3_resized], dim=1)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        e2_resized = nn.functional.interpolate(e2, size=d2.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, e2_resized], dim=1)
        d2 = self.dec2(d2)

        output = self.final_conv(d2)
        return nn.functional.interpolate(output, size=(70, 70), mode='bilinear', align_corners=False)


class AttentionGate(nn.Module):
    """
    An Attention Gate module, as described in the Attention U-Net paper.
    This gate learns to focus on salient features from the skip connection.
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class UNetWithAttention(nn.Module):
    """
    A U-Net architecture that incorporates Attention Gates in the skip connections.
    """
    def __init__(self, in_channels=5, out_channels=1):
        super(UNetWithAttention, self).__init__()

        # --- Encoder ---
        self.enc1 = conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.enc2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Bottleneck ---
        self.bottleneck = conv_block(256, 512)

        # --- Decoder ---
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.Att3 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.dec3 = conv_block(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.Att2 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.dec2 = conv_block(256, 128)

        # --- Final Output ---
        self.final_conv = nn.Conv2d(128, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        # Bottleneck
        b = self.bottleneck(p3)

        # Decoder with Attention
        d3 = self.upconv3(b)
        e3_resized = nn.functional.interpolate(e3, size=d3.shape[2:], mode='bilinear', align_corners=False)
        x3_att = self.Att3(g=d3, x=e3_resized)
        d3 = torch.cat([x3_att, d3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        e2_resized = nn.functional.interpolate(e2, size=d2.shape[2:], mode='bilinear', align_corners=False)
        x2_att = self.Att2(g=d2, x=e2_resized)
        d2 = torch.cat([x2_att, d2], dim=1)
        d2 = self.dec2(d2)

        output = self.final_conv(d2)
        return nn.functional.interpolate(output, size=(70, 70), mode='bilinear', align_corners=False)


class Discriminator(nn.Module):
    """
    A simple PatchGAN-style discriminator for the GAN feasibility test.
    It takes a 70x70 velocity map and determines if it is real or fake.
    The architecture is a series of downsampling convolutional blocks.
    """
    def __init__(self, in_channels=1):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns downsampling layers of each discriminator block."""
            block = [nn.Conv2d(in_filters, out_filters, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)]
            if bn:
                block.append(nn.InstanceNorm2d(out_filters))
            return block

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, bn=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img):
        return self.model(img)


class ForwardModel(nn.Module):
    """
    A model for the FORWARD problem (Physics Simulation).

    Task: Transform a velocity map back into seismic data.
    Input Shape:  (Batch, 1, 70, 70)
    Output Shape: (Batch, 5, 1000, 70)

    This model's purpose is to enforce physical plausibility. The U-Net
    predicts a velocity map. We feed that map into this model. If the
    predicted map was physically correct, this model should be able to
    reconstruct the original seismic data from it. Any failure to do so
    is penalized by the "cycle consistency loss".
    """

    def __init__(self, in_channels=1, out_channels=5):
        super(ForwardModel, self).__init__()
        # The architecture here is also a guess and would need significant
        # design effort. It's essentially a generator network that would
        # use transposed convolutions to upsample the 70x70 map to 1000x70.
        self.main = nn.Sequential(
            conv_block(in_channels, 64),
            # ... layers to expand dimensions and features ...
            nn.Conv2d(64, out_channels, kernel_size=1),
        )

    def forward(self, velocity_map):
        # This is a placeholder. A real model would need transposed convolutions
        # or other upsampling methods to generate the (1000, 70) output shape.
        reconstructed_seismic_placeholder = self.main(velocity_map)
        
        # We use interpolation to force the output shape for this skeleton.
        return nn.functional.interpolate(reconstructed_seismic_placeholder, size=(1000, 70), mode='bilinear', align_corners=False)


# --- Transformer Building Blocks ---

class PatchEmbedding(nn.Module):
    """
    Converts a 2D image into a 1D sequence of patch embeddings.
    Input: (B, C, H, W) -> (B, 5, 1000, 70)
    Output: (B, N, D) -> (B, num_patches, embed_dim)
    """
    def __init__(self, in_channels=5, patch_size=(20, 7), embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/patch_h, W/patch_w)
        x = x.flatten(2) # (B, embed_dim, num_patches)
        x = x.transpose(1, 2) # (B, num_patches, embed_dim)
        return x

class TransformerBlock(nn.Module):
    """
    A standard Transformer block with Multi-Head Self-Attention and an MLP.
    """
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Self-attention part
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        # MLP part
        x = x + self.mlp(self.norm2(x))
        return x

# --- End Transformer Building Blocks ---

class TransformerGenerator(nn.Module):
    """
    A Vision Transformer (ViT) based generator.
    It uses a Transformer Encoder to process the image as a sequence of patches
    and a convolutional Decoder to reconstruct the output velocity map.
    """
    def __init__(self, in_channels=5, out_channels=1, patch_size=(20, 7), 
                 embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        
        # 1. Patch and Position Embedding
        self.patch_embed = PatchEmbedding(in_channels, patch_size, embed_dim)
        # H/patch_h = 1000/20 = 50. W/patch_w = 70/7 = 10.
        num_patches = (1000 // patch_size[0]) * (70 // patch_size[1])
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        # 2. Transformer Encoder
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # 3. Convolutional Decoder
        # Takes the sequence from the encoder and reconstructs the image.
        # The decoder needs to upsample from the patch-level resolution.
        # Patch grid: (50, 10)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 256, kernel_size=3, stride=2, padding=1, output_padding=1), # -> 100x20
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=(1,0)), # -> 200x39
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), # -> 400x78
            nn.ReLU(inplace=True),
            # Final convolution to get to the right size and channels
            nn.Conv2d(64, out_channels, kernel_size=7, stride=1, padding=3) # -> 400x78
        )

    def forward(self, x):
        # --- Encoder ---
        x = self.patch_embed(x)
        x = x + self.pos_embed
        
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        # --- Decoder ---
        # Reshape sequence back into a 2D feature map
        # (B, num_patches, embed_dim) -> (B, embed_dim, H/patch_h, W/patch_w)
        h = 1000 // self.patch_embed.patch_size[0]
        w = 70 // self.patch_embed.patch_size[1]
        x = x.transpose(1, 2).reshape(x.shape[0], x.shape[2], h, w)
        
        x = self.decoder(x)
        
        # Interpolate to the final target size to ensure exact dimensions
        return nn.functional.interpolate(x, size=(70, 70), mode='bilinear', align_corners=False)