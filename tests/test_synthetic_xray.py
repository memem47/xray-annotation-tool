from src.synthetic_xray import generate_phantom

def test_size_and_dtype():
    img = generate_phantom(size=256)
    assert img.shape == (256, 256)
    assert img.dtype == "uint8"