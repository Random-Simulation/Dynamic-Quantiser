"""Generate dynamic-quantiser.ico: a large "DQ" monogram (Q overlapped into
the D) that fills the 256x256 tile, filled with a purple->blue gradient, on a
transparent background. Run with the build_env python.

Tunables:
  OVERLAP_150  how far the Q slides into the D, measured at font size 150
               (scales with render size to keep the same look)
  FILL         fraction of the canvas the lockup should fill (larger dim)
  C1 / C2      gradient end colours (top-left -> bottom-right)
"""
import sys
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W = 256
OUT = "dynamic-quantiser.ico"
OVERLAP_150 = 65          # approved overlap, at font size 150
FILL = 0.94               # larger dimension as a fraction of W
RFS = 200                 # render font size (crisp, then fine-scale to fill)
RC = 640                  # render canvas size
C1 = (16, 12, 40)        # near-black blue-purple (top-left)
C2 = (92, 58, 158)       # darkish purple (bottom-right)
HALO_COL = (155, 45, 255)  # purple halo colour
HALO_BLUR = 1            # halo softness (px)
HALO_ALPHA = 1.0         # halo opacity (0..1)


def load_font():
    candidates = [
        r"C:\Windows\Fonts\arialbl.ttf",   # Arial Black
        r"C:\Windows\Fonts\arialbd.ttf",   # Arial Bold
        r"C:\Windows\Fonts\segoeuib.ttf",  # Segoe UI Bold
        r"C:\Windows\Fonts\impact.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    ]
    for c in candidates:
        try:
            return ImageFont.truetype(c, RFS)
        except Exception:
            continue
    sys.exit("No suitable TTF font found in C:\\Windows\\Fonts")


def main():
    font = load_font()
    overlap = round(OVERLAP_150 * RFS / 150)

    # --- render the DQ lockup (white) on a roomy canvas ---
    cv = Image.new("RGBA", (RC, RC), (0, 0, 0, 0))
    d = ImageDraw.Draw(cv)
    db = font.getbbox("D")
    qb = font.getbbox("Q")
    dw, qw = db[2] - db[0], qb[2] - qb[0]
    x0 = (RC - (dw - overlap + qw)) // 2
    top_off = min(db[1], qb[1])
    bot_off = max(db[3], qb[3])
    Y = RC // 2 - (top_off + bot_off) // 2
    d.text((x0 - db[0], Y), "D", font=font, fill=(255, 255, 255, 255))
    d.text((x0 + dw - overlap - qb[0], Y), "Q", font=font, fill=(255, 255, 255, 255))

    # --- crop to tight ink box, scale to fill, centre ---
    lockup = cv.crop(cv.getchannel("A").getbbox())
    lw, lh = lockup.size
    scale = (W * FILL) / max(lw, lh)
    nw, nh = round(lw * scale), round(lh * scale)
    lockup = lockup.resize((nw, nh), Image.LANCZOS)

    shape = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    shape.alpha_composite(lockup, ((W - nw) // 2, (W - nh) // 2))

    # --- diagonal purple->blue gradient, letters as the alpha ---
    grad = Image.new("RGBA", (W, W))
    gpx = grad.load()
    for y in range(W):
        for x in range(W):
            t = (x + y) / (2 * (W - 1))
            gpx[x, y] = (round(C1[0] + (C2[0] - C1[0]) * t),
                         round(C1[1] + (C2[1] - C1[1]) * t),
                         round(C1[2] + (C2[2] - C1[2]) * t), 255)
    grad.putalpha(shape.getchannel("A"))

    # --- subtle purple halo behind the letters ---
    halo_mask = shape.getchannel("A").filter(ImageFilter.GaussianBlur(HALO_BLUR))
    halo_mask = halo_mask.point(lambda a: int(a * HALO_ALPHA))
    halo = Image.new("RGBA", (W, W), HALO_COL + (255,))
    halo.putalpha(halo_mask)

    final = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    final.alpha_composite(halo)
    final.alpha_composite(grad)

    final.save(OUT, sizes=[(16, 16), (24, 24), (32, 32),
                           (48, 48), (64, 64), (128, 128), (256, 256)])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
