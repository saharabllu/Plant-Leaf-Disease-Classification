# Dataset Description
The dataset used in this project is provided by the **Informatics Festival (I-Fest) 6.0** committee. It is a comprehensive image dataset containing photographs of plant leaves taken in various conditions and backgrounds.

## Dataset Characteristics
Before modeling, Exploratory Data Analysis (EDA) was performed to understand the data distribution. The dataset presents a significant class imbalance, which poses a challenge for model training.

- **Total Images:** 4,005
- **Total Classes:** 27 (Combining plant types and specific diseases)

**Detailed Distribution per Plant:**
- **Potato (1,152 images):** Early Blight (500), Late Blight (500), Healthy (152).
- **Lemon (1,354 images):** 9 classes including Anthracnose, Bacterial Blight, Citrus Canker, Healthy, etc. (ranging from 100 to 210 images per class).
- **Tea (500 images):** Algal leaf (113), Bird eye spot (100), Brown blight (113), Gray blight (100), Healthy (74).
- **Tomato (999 images):** 10 classes including Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Healthy, etc. (mostly exactly 100 images per class).

*Note: The imbalance is evident, for instance, Potato Early Blight has 500 images, while Tea Healthy only has 74.*

Due to the large file size (851 MB), the raw dataset (`archive_folder.zip`) is not hosted directly in this repository. You can download the dataset via the following Google Drive link:
**[https://drive.google.com/file/d/13TMJmmNl1FopDFthOObtItm4KMdJWD4h/view?usp=drive_link]**
