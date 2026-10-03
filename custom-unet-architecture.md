# Custom UNet Architecture — Lane Segmentation

Single-class (lane) binary segmentation network. Input `3x48x48` RGB, output
`1x48x48` logits (apply sigmoid + threshold for a binary mask). Implementation:
[model.py](model.py).

- Base channel width: 32 (doubles each downsample: 32 → 64 → 128 → 256 → 512)
- 4 encoder stages + bottleneck, 4 decoder stages with skip connections
- Downsample: `MaxPool2x2`; Upsample: `ConvTranspose2x2`
- Each conv stage is a `DoubleConv`: `(Conv3x3 → BatchNorm → ReLU) x2`

```mermaid
flowchart TD
    IN["Input<br/>3 x 48 x 48"] --> ENC1

    subgraph Encoder
        ENC1["DoubleConv<br/>3 to 32<br/>48x48"]
        P1["MaxPool 2x2"]
        ENC2["DoubleConv<br/>32 to 64<br/>24x24"]
        P2["MaxPool 2x2"]
        ENC3["DoubleConv<br/>64 to 128<br/>12x12"]
        P3["MaxPool 2x2"]
        ENC4["DoubleConv<br/>128 to 256<br/>6x6"]
        P4["MaxPool 2x2"]
    end

    ENC1 --> P1 --> ENC2 --> P2 --> ENC3 --> P3 --> ENC4 --> P4

    P4 --> BN["Bottleneck DoubleConv<br/>256 to 512<br/>3x3"]

    subgraph Decoder
        UP1["ConvTranspose 2x2<br/>512 to 256<br/>-> 6x6"]
        CAT1["Concat with ENC4<br/>256 + 256 = 512"]
        DEC1["DoubleConv<br/>512 to 256<br/>6x6"]

        UP2["ConvTranspose 2x2<br/>256 to 128<br/>-> 12x12"]
        CAT2["Concat with ENC3<br/>128 + 128 = 256"]
        DEC2["DoubleConv<br/>256 to 128<br/>12x12"]

        UP3["ConvTranspose 2x2<br/>128 to 64<br/>-> 24x24"]
        CAT3["Concat with ENC2<br/>64 + 64 = 128"]
        DEC3["DoubleConv<br/>128 to 64<br/>24x24"]

        UP4["ConvTranspose 2x2<br/>64 to 32<br/>-> 48x48"]
        CAT4["Concat with ENC1<br/>32 + 32 = 64"]
        DEC4["DoubleConv<br/>64 to 32<br/>48x48"]
    end

    BN --> UP1 --> CAT1 --> DEC1
    DEC1 --> UP2 --> CAT2 --> DEC2
    DEC2 --> UP3 --> CAT3 --> DEC3
    DEC3 --> UP4 --> CAT4 --> DEC4

    DEC4 --> OUTC["Conv 1x1<br/>32 to 1"]
    OUTC --> OUT["Output logits<br/>1 x 48 x 48"]

    ENC4 -. skip .-> CAT1
    ENC3 -. skip .-> CAT2
    ENC2 -. skip .-> CAT3
    ENC1 -. skip .-> CAT4
```

## Layer-by-layer summary

| Stage      | Operation                     | Output shape (C x H x W) |
|------------|--------------------------------|---------------------------|
| Input      | —                               | 3 x 48 x 48                |
| Encoder 1  | DoubleConv(3, 32)                | 32 x 48 x 48                |
| Encoder 2  | MaxPool + DoubleConv(32, 64)     | 64 x 24 x 24                |
| Encoder 3  | MaxPool + DoubleConv(64, 128)    | 128 x 12 x 12                |
| Encoder 4  | MaxPool + DoubleConv(128, 256)   | 256 x 6 x 6                  |
| Bottleneck | MaxPool + DoubleConv(256, 512)   | 512 x 3 x 3                  |
| Decoder 1  | ConvTranspose(512,256) + concat(Enc4) + DoubleConv(512,256) | 256 x 6 x 6 |
| Decoder 2  | ConvTranspose(256,128) + concat(Enc3) + DoubleConv(256,128) | 128 x 12 x 12 |
| Decoder 3  | ConvTranspose(128,64) + concat(Enc2) + DoubleConv(128,64)   | 64 x 24 x 24 |
| Decoder 4  | ConvTranspose(64,32) + concat(Enc1) + DoubleConv(64,32)     | 32 x 48 x 48 |
| Output     | Conv1x1(32, 1)                   | 1 x 48 x 48 (logits)          |

Total parameters: 7,763,041 (printed by running `python model.py`).
