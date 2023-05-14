
# Data Augmentation Techniques

This code performs data augmentation on images of the COCO datset and their corresponding annotations. The augmentation methods provided include flipping, scaling, and noise injection.

&nbsp;






## Requirements:

 - python
 - torch
 - torchvision
 - PIL
 - json

 &nbsp;




## Running Tests

To run the code use following command:

```bash
  python main.py --task <task_type> --transformer <augmentation_type>
```
Replace <task_type> with either "all" to apply the augmentation to the entire dataset, or "example" to apply it to the first 10 samples of the dataset

<augmentation_type> specifies the type of augmentation you want to perform. You can replace it to "flip", "scale", or "noise_injection"

For example, if you want to augment one sample as an example with horizontal flip augmentation, you would run the command:

```bash
  python main.py --task example --transformer flip
```
Or if you want to augment the entire dataset with noise injection augmentation, you would run the command:

```bash
  python main.py --task all --transformer noise_injection
```
#### Note:
To avoid downloading the dataset again after the initial run, you can set the optional argument download to 'false'.
The default value for this argument is 'true'.

example:

```bash
  python main.py --task all --transformer noise_injection --download false
```
&nbsp;

## Output
The results can be found in two folders named "augmented_images" and "augmented_annotation", which are located in the same directory as the running code