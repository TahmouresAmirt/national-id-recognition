# Iranian National ID Recognition System

A robust computer vision system for extracting and validating Iranian National ID numbers from ID card images.

This project focuses on practical, real-world scenarios and is designed to work with unconstrained images captured under varying lighting conditions and orientations.

---

## Key Features

- Automatic detection of ID card and National ID region
- Orientation-aware image normalisation
- Robust digit extraction and recognition
- Confidence-based result validation
- Built-in Iranian National ID checksum verification
- Simple local application interface

---

## Example

### Input
An image of an Iranian National ID card:

![Sample Input](sample_input.jpg)

### Output
```json
{
  "code": "2980231002",
  "checksum_ok": true,
  "mean_conf": 0.99,
  "min_conf": 0.95
}
