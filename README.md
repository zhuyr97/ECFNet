# 🥇 Winner solution on the MIPI 2022 Challenge on Under-display Camera Image Restoration

Our team (USTC_WXYZ) wins the [MIPI 2022 Challenge on Under-display Camera Image Restoration](https://mipi-challenge.org/)!

## Dependencies

- Python
- Pytorch (1.8+)
- Numpy

## Testing on the Challendge Dataset
1. Prepare the testing data:  put the testing data into './test_data' folder.

2. To test our model , run the command:   "python testing.py"

3. Then, the processed results will be saved in './results' folder.

**Note that the pre-trained model has been saved in './ckpt' folder. And the pre-trained model is trained on the datasets privided by the challenge.

The testing results of the MIPI 2022 Challenge also could be downloaded from [here](https://drive.google.com/drive/folders/1KqWik69-YI9-K352kwKSY8M-gQnNN0L9?usp=sharing).

## Testing on the OTLED Dataset

We further training our scaled model on the [OTLED dataset](https://github.com/JaihyunKoh/BNUDC). The corresponding results could be found in [here](https://drive.google.com/drive/folders/1BL1vbb0PPOKom1iTYcf_HWFLy0ROF9ay?usp=sharing).




Thanks to the great efforts of the open-sourced projects [MIMOUNet](https://github.com/chosj95/MIMO-UNet).

