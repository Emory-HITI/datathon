import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import pad
from PIL import Image
import pandas as pd
import numbers


def get_padding(img):
    w, h = img.size
    max_wh = np.max([w, h])
    h_padding = (max_wh - w) / 2
    v_padding = (max_wh - h) / 2
    l_pad = h_padding if h_padding % 1 == 0 else h_padding + 0.5
    t_pad = v_padding if v_padding % 1 == 0 else v_padding + 0.5
    r_pad = h_padding if h_padding % 1 == 0 else h_padding - 0.5
    b_pad = v_padding if v_padding % 1 == 0 else v_padding - 0.5
    padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))
    return padding


class NewPad(object):
    def __init__(self, fill=0, padding_mode='constant'):
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        return pad(img, list(get_padding(img)), self.fill, self.padding_mode)

    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1}, padding_mode={2})'. \
            format(self.fill, self.padding_mode)


data_transforms = {
    'train': transforms.Compose([
        NewPad(),
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        NewPad(),
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        NewPad(),
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


class EMBED(Dataset):
    def __init__(self,
                 csv_path,
                 target_labels=None,
                 sensitive_attribute="ETHNICITY",
                 shuffle=True,
                 transform=None,
                 mode='train'):
        self.sensitive_attributes = sensitive_attribute
        self.target_labels = target_labels if isinstance(target_labels, list) else [target_labels]
        self.transform = transform
        self.mode = mode

        if isinstance(self.sensitive_attributes, str):
            self.sensitive_attributes = [self.sensitive_attributes]

        self.df = pd.read_csv(csv_path)

        self.images_list = self.df["png_path"].tolist()

        self.num_imgs = len(self.df)

        # print("SELF ATTRIBUTES")
        # print(self.sensitive_attributes)

        # for sa in self.sensitive_attributes:
        #     if sa == "":
        #         self.df[sa] = self.df[sa].apply(lambda x: 1 if "Caucasian or White" in x else 0)
        #     elif sa == "age_at_study":
        #         self.df[sa] = self.df[sa].apply(lambda x: 1 if x > 60 else 0)

        if shuffle:
            data_index = list(range(self.num_imgs))
            np.random.shuffle(data_index)
            self.df = self.df.iloc[data_index]

        # self.images_list = [
        #     os.path.
        # ]
        self.targets = self.df[self.target_labels].values.tolist()

        self.a_dict = {}
        for attribute in self.sensitive_attributes:
            self.a_dict[attribute] = self.df[attribute].values.tolist()

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        image = Image.open(self.images_list[idx]).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        target = torch.tensor(self.targets[idx]).view(-1).long()

        a = {}

        for k, v in self.a_dict.items():
            a[k] = torch.tensor(v[idx]).view(-1).long()

        return image, target, a


if __name__ == '__main__':
    import numpy as np

    im = Image.fromarray(np.ones((333, 4034)))

    t_img = data_transforms["train"](im)
    print(t_img.shape == torch.Size([3, 224, 224]))
