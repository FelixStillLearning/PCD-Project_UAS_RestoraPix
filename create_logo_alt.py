import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# Direktori untuk menyimpan logo
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'Icon RestoraPix')
os.makedirs(output_dir, exist_ok=True)

# Ukuran logo
width, height = 512, 512
img = Image.new('RGBA', (width, height), color=(255, 255, 255, 0))
draw = ImageDraw.Draw(img)

# Warna
primary_color = (25, 55, 109)  # Biru tua
accent_color = (255, 140, 0)   # Oranye
text_color = (255, 255, 255)   # Putih

# Buat lingkaran luar
margin = 50
draw.ellipse([(margin, margin), (width-margin, height-margin)], fill=primary_color)

# Buat simbol desain - split lingkaran untuk restorasi
# Gambar lingkaran "sebelum" - abu-abu
split_margin = 120
draw.pieslice(
    [(split_margin, split_margin), (width-split_margin, height-split_margin)],
    start=180, end=360, 
    fill=(150, 150, 150)
)

# Gambar lingkaran "sesudah" - warna
draw.pieslice(
    [(split_margin, split_margin), (width-split_margin, height-split_margin)],
    start=0, end=180, 
    fill=accent_color
)

# Garis pembagi di tengah
line_points = [(width//2, split_margin), (width//2, height-split_margin)]
draw.line(line_points, fill=(255, 255, 255), width=5)

# Tambahkan teks "R" di tengah
font_size = 150
try:
    font = ImageFont.load_default()
    # Coba gunakan font default dengan ukuran lebih besar
    font = ImageFont.truetype("arial.ttf", font_size)
except Exception:
    # Jika tidak ada font, tetap gunakan default
    pass

# Posisikan teks di tengah
text = "R"
text_x = width // 2 - font_size // 3
text_y = height // 2 - font_size // 1.5
draw.text((text_x, text_y), text, font=font, fill=text_color)

# Tambahkan teks "RestoraPix" di bawah
try:
    font_small = ImageFont.load_default()
    font_small = ImageFont.truetype("arial.ttf", 40)
except Exception:
    pass

brand_text = "RestoraPix"
brand_width = 200  # Perkiraan
brand_x = (width - brand_width) // 2
brand_y = height - 120
draw.text((brand_x, brand_y), brand_text, font=font_small, fill=text_color)

# Simpan logo
logo_path = os.path.join(output_dir, 'logo_restorapix_alt.png')
img.save(logo_path)

print(f"Logo alternatif berhasil dibuat dan disimpan di: {logo_path}")
