dataset_type = 'PdfDataset'
data_root = '/data/yexiaoyu/pdf-image-dataset_v3_example/English'

train = dict(
    type=dataset_type,
    ann_file=f'{data_root}/instances_training.json',
    img_prefix=f'{data_root}/imgs',
    pipeline=None)

test = dict(
    type=dataset_type,
    ann_file=f'{data_root}/COCO_pdf_image_dataset_v3_example.json',
    img_prefix=f'{data_root}/Images',
    pipeline=None)

train_list = [train]

test_list = [test]
