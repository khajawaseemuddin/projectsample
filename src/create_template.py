from PIL import Image, ImageDraw, ImageFont
import os

def create_challan_template():
    # Create template directory if it doesn't exist
    os.makedirs("templates", exist_ok=True)
    
    # Create a blank white image
    width, height = 800, 600
    template = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(template)
    
    # Add header
    header_font = ImageFont.truetype("arial.ttf", 40)
    draw.text((width//2 - 100, 50), "TRAFFIC CHALLAN", fill="black", font=header_font)
    
    # Add fields
    field_font = ImageFont.truetype("arial.ttf", 20)
    fields = [
        ("Challan ID:", (50, 150)),
        ("Date:", (50, 200)),
        ("Time:", (400, 200)),
        ("Maximum Speed:", (50, 250)),
        ("License Plate:", (50, 300)),
        ("Vehicle ID:", (50, 350)),
        ("Location:", (50, 400)),
        ("Fine Amount:", (50, 450))
    ]
    
    for field, pos in fields:
        draw.text(pos, field, fill="black", font=field_font)
    
    # Add footer
    footer_font = ImageFont.truetype("arial.ttf", 15)
    draw.text((width//2 - 200, height - 50), 
              "Please pay the fine within 7 days to avoid additional penalties", 
              fill="black", font=footer_font)
    
    # Save template
    template.save("templates/challan_template.jpg")
    print("Challan template created successfully!")

if __name__ == "__main__":
    create_challan_template() 