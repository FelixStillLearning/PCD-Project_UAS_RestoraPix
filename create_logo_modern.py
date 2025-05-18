import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# Direktori untuk menyimpan logo
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'Icon RestoraPix')
os.makedirs(output_dir, exist_ok=True)

# Ukuran logo
width, height = 512, 512
img = Image.new('RGBA', (width, height), color=(0, 0, 0, 0))
draw = ImageDraw.Draw(img)

# Warna
blue = (0, 120, 215)  # Biru Microsoft-style
orange = (255, 140, 0)  # Oranye untuk aksen
dark_blue = (0, 70, 140)  # Biru gelap untuk bayangan

# Buat lingkaran sebagai background
margin = 60
draw.ellipse(
    [(margin, margin), (width - margin, height - margin)],
    fill=blue
)

# Buat lingkaran dengan efek kamera
center_x, center_y = width // 2, height // 2 - 30
circle_radius = 90

# Lingkaran luar
draw.ellipse(
    [(center_x - circle_radius, center_y - circle_radius),
     (center_x + circle_radius, center_y + circle_radius)],
    fill=(255, 255, 255),
    outline=orange,
    width=5
)

# Lingkaran dalam (lensa)
inner_radius = circle_radius - 20
draw.ellipse(
    [(center_x - inner_radius, center_y - inner_radius),
     (center_x + inner_radius, center_y + inner_radius)],
    fill=orange
)

# Efek cahaya di lensa
highlight_radius = inner_radius // 3
highlight_offset = inner_radius // 2
draw.ellipse(
    [(center_x - highlight_radius - highlight_offset, center_y - highlight_radius - highlight_offset),
     (center_x - highlight_offset, center_y - highlight_offset)],
    fill=(255, 255, 255, 180)
)

# Emblem "RP" di bawah lingkaran
emblem_y = center_y + circle_radius + 40
try:
    font = ImageFont.truetype("arial.ttf", 120)
except Exception:
    font = ImageFont.load_default()

emblem_text = "RP"
emblem_x = center_x - 60  # Perkiraan posisi

draw.text((emblem_x, emblem_y), emblem_text, font=font, fill=(255, 255, 255))

# Tambahkan teks "RestoraPix" di bagian bawah
try:
    footer_font = ImageFont.truetype("arial.ttf", 36)
except Exception:
    footer_font = ImageFont.load_default()

footer_text = "RestoraPix"
footer_y = height - margin - 60
footer_x = center_x - 90  # Perkiraan posisi

draw.text((footer_x, footer_y), footer_text, font=footer_font, fill=(255, 255, 255))

# Simpan logo
logo_path = os.path.join(output_dir, 'logo_restorapix_modern.png')
img.save(logo_path)

print(f"Logo modern berhasil dibuat dan disimpan di: {logo_path}")

# Lingkaran dalam (lensa)
inner_radius = circle_radius - 20
draw.ellipse(
    [(center_x - inner_radius, center_y - inner_radius),
     (center_x + inner_radius, center_y + inner_radius)],
    fill=orange
)

# Efek cahaya di lensa
highlight_radius = inner_radius // 3
highlight_offset = inner_radius // 2
draw.ellipse(
    [(center_x - highlight_radius - highlight_offset, center_y - highlight_radius - highlight_offset),
     (center_x - highlight_offset, center_y - highlight_offset)],
    fill=(255, 255, 255, 180)
)

# Emblem "RP" di bawah lingkaran
emblem_y = center_y + circle_radius + 40
try:
    font = ImageFont.truetype("arial.ttf", 120)
except Exception:
    font = ImageFont.load_default()

emblem_text = "RP"
emblem_x = center_x - 60  # Perkiraan posisi

draw.text((emblem_x, emblem_y), emblem_text, font=font, fill=(255, 255, 255))

# Tambahkan teks "RestoraPix" di bagian bawah
try:
    footer_font = ImageFont.truetype("arial.ttf", 36)
except Exception:
    footer_font = ImageFont.load_default()

footer_text = "RestoraPix"
footer_y = height - margin - 60
footer_x = center_x - 90  # Perkiraan posisi

draw.text((footer_x, footer_y), footer_text, font=footer_font, fill=(255, 255, 255))

# Simpan logo
logo_path = os.path.join(output_dir, 'logo_restorapix_modern.png')
img.save(logo_path)

print(f"Logo modern berhasil dibuat dan disimpan di: {logo_path}")
