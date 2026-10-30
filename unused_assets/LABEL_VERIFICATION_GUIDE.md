# Label Verification Guide

## Three ways to check if labels are correct:

### 1. **Quick Statistics Report** (Fastest)
Run this to see label distribution and statistics:

```bash
python verify_labels.py
```

**Output includes:**
- Total number of labeled images
- Distribution across 4 classes
- Average confidence per class
- Low-confidence images (< 0.30 confidence)
- Sample images with highest confidence per category

---

### 2. **Visual GUI Review Tool** (Recommended)
Review images one-by-one with a GUI interface:

```bash
python review_gui.py
```

**Features:**
- Browse images with assigned labels
- View confidence scores
- Correct labels if needed (radio buttons)
- Save corrections back to CSV
- Navigation: Previous/Next buttons

---

### 3. **Manual CSV Review** (For spreadsheet lovers)
Open and review the CSV file directly:

- File: `image_labels_clean.csv`
- Columns: `image_path | label | class_name | confidence`
- Edit labels directly in Excel/CSV editor
- Save and use for training

---

## Quality Assessment

**Check these metrics:**

| Metric | Good | Poor |
|--------|------|------|
| **Confidence Average** | > 0.35 | < 0.30 |
| **Label Distribution** | Balanced | Highly skewed |
| **Visual Accuracy** | 80%+ visually correct | < 50% |

---

## If Labels Look Wrong:

1. **Low confidence overall (< 0.30)?**
   - CLIP may not be ideal for dental X-rays
   - Manually label 50-100 images as a training set
   - Train a custom classifier on those

2. **Wrong categories assigned?**
   - Use `review_gui.py` to batch correct them
   - Or manually edit `image_labels_clean.csv`

3. **Need better accuracy?**
   - Combine CLIP labels with manual review
   - Create a hybrid dataset

---

## Next Steps After Verification:

1. ✓ Review labels with one of the 3 methods above
2. ✓ Make corrections if needed
3. ✓ Update `image_labels_clean.csv` with final labels
4. ✓ Create training dataset structure:
   ```
   dataset/
   ├── train/
   │   ├── 0_no_treatment/
   │   ├── 1_incomplete/
   │   ├── 2_complete/
   │   └── 3_failure/
   └── val/
       └── (same structure)
   ```
5. ✓ Train classifier on the labeled data

---

## File Locations

- **Labels:** `image_labels.csv` (raw) → `image_labels_clean.csv` (cleaned/deduplicated)
- **Images:** `dataset/images/`
- **Tools:** 
  - `verify_labels.py` - Statistics
  - `review_gui.py` - Visual review
  - `image_labels_clean.csv` - Edit manually

