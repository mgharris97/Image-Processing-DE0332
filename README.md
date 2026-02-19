# Image Processing DE0332 Lab 1 (Image Blending Modes & Transition Effects)

This lab implements several image blending modes and a transparency (opacity) transition in Python using NumPy for all blending math. OpenCV is used only for image I/O (loading/saving).

## Features

Implemented blending modes:

	- Multiply
  
	- Linear Burn (listed as Subtraction / Linear Burn in the slides)
  
	- Color Dodge
  
	- Transparency (Opacity blending): C = dA + (1-d)B

All computations are performed in float32 for arithmetic flexibility and then clamped to [0, 255] and converted back to uint8.

## Project Structure

```bash
.
├── main.py
├── src/
│   ├── multiply.py
│   ├── linear_burn.py
│   ├── dodge.py
│   └── transparency.py
├── Results/            
├── README.md
└── .gitignore
```

## Reqirements
- Python 3
- NumPy
- OpenCV (v2)

## Install dependencies 
```python
pip install numpy opencv-python
```

## CLI Usage
```bash
python3 main.py --img1 /path/to/image1.jpg --img2 /path/to/image2.jpg --mode multiply
```

## Available blending modes
- multiply
- linear_burn
- didge
- transparency

## Example CLI commands
```bash
# Multiply
python3 main.py --img1 a.jpg --img2 b.jpg --mode multiply

# Linear Burn
python3 main.py --img1 a.jpg --img2 b.jpg --mode linear_burn

# Color Dodge
python3 main.py --img1 a.jpg --img2 b.jpg --mode dodge

# Transparency (opacity factor d in [0,1])
python3 main.py --img1 a.jpg --img2 b.jpg --mode transparency --d 0.35
```

## Outputs
By default output is saved to a Results/ folder however,  output folder can be changed:
```bash
python3 main.py --img1 a.jpg --img2 b.jpg --mode multiply --out MyOutputs
```


Matthew Harris

241ADB166

Feb 19, 2026


