from dataset import SegmentationDataset

ds = SegmentationDataset(
    "dataset/train/images",
    "dataset/train/masks",
    debug=True
)

_ = ds[0]