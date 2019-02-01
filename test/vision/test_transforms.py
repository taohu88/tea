import torch
import tea.vision.cv.transforms as transforms
import numpy as np
from PIL import Image
import unittest
import random


class Tester(unittest.TestCase):

    def test_crop(self):
        height = random.randint(10, 32) * 2
        width = random.randint(10, 32) * 2
        oheight = random.randint(4, (height - 2) / 2) * 2
        owidth = random.randint(4, (width - 2) / 2) * 2

        img = torch.ones(3, height, width)
        oh1 = (height - oheight) // 2
        ow1 = (width - owidth) // 2
        imgnarrow = img[:, oh1:oh1 + oheight, ow1:ow1 + owidth]
        imgnarrow.fill_(0)
        result = transforms.Compose([
            transforms.TensorToNumpy(),
            transforms.CenterCrop((oheight, owidth)),
            transforms.ToTensor(),
        ])(img)
        print(result.shape)
        assert((oheight == result.size()[1]) and (owidth == result.size()[2]))
        assert result.sum() == 0, "height: " + str(height) + " width: " \
                                  + str(width) + " oheight: " + str(oheight) + " owidth: " + str(owidth)
        oheight += 1
        owidth += 1
        result = transforms.Compose([
            transforms.TensorToNumpy(),
            transforms.CenterCrop((oheight, owidth)),
            transforms.ToTensor(),
        ])(img)
        sum1 = result.sum()
        # top left is 1.0 now
        assert(result[0,0,0] >= 1.0)
        assert sum1 > 1, "height: " + str(height) + " width: " \
                         + str(width) + " oheight: " + str(oheight) + " owidth: " + str(owidth)
        oheight += 1
        owidth += 1
        result = transforms.Compose([
            transforms.TensorToNumpy(),
            transforms.CenterCrop((oheight, owidth)),
            transforms.ToTensor(),
        ])(img)
        # bottom right is 1.0 now.
        assert(result[0,oheight-1, owidth-1] >= 1.0)
        sum2 = result.sum()
        assert sum2 > 0, "height: " + str(height) + " width: " \
                         + str(width) + " oheight: " + str(oheight) + " owidth: " + str(owidth)
        assert sum2 > sum1, "height: " + str(height) + " width: " \
                            + str(width) + " oheight: " + str(oheight) + " owidth: " + str(owidth)

    def test_five_crop(self):
        to_np_image = transforms.TensorToNumpy()

        h = random.randint(5, 25)
        w = random.randint(5, 25)
        for single_dim in [True, False]:
            crop_h = random.randint(1, h)
            crop_w = random.randint(1, w)
            if single_dim:
                crop_h = min(crop_h, crop_w)
                crop_w = crop_h
                transform = transforms.FiveCrop(crop_h)
            else:
                transform = transforms.FiveCrop((crop_h, crop_w))

            img = torch.FloatTensor(3, h, w).uniform_()
            results = transform(to_np_image(img))

            assert len(results) == 5
            for crop in results:
                assert crop.shape[:2] == (crop_h, crop_w)

            to_np_image = transforms.TensorToNumpy()
            tl = to_np_image(img[:, 0:crop_h, 0:crop_w])
            tr = to_np_image(img[:, 0:crop_h, w - crop_w:])
            bl = to_np_image(img[:, h - crop_h:, 0:crop_w])
            br = to_np_image(img[:, h - crop_h:, w - crop_w:])
            center = transforms.CenterCrop((crop_h, crop_w))(to_np_image(img))
            expected_output = (tl, tr, bl, br, center)
            for o, e in zip(results, expected_output):
                assert (o == e).all()

    def test_ten_crop(self):
        to_np_image = transforms.TensorToNumpy()

        h = random.randint(5, 25)
        w = random.randint(5, 25)
        for should_vflip in [True, False]:
            for single_dim in [True, False]:
                crop_h = random.randint(1, h)
                crop_w = random.randint(1, w)
                if single_dim:
                    crop_h = min(crop_h, crop_w)
                    crop_w = crop_h
                    transform = transforms.TenCrop(crop_h,
                                                   vertical_flip=should_vflip)
                    five_crop = transforms.FiveCrop(crop_h)
                else:
                    transform = transforms.TenCrop((crop_h, crop_w),
                                                   vertical_flip=should_vflip)
                    five_crop = transforms.FiveCrop((crop_h, crop_w))

                tensor_img = torch.FloatTensor(3, h, w).uniform_()
                img = to_np_image(tensor_img)

                results = transform(img)
                expected_output = five_crop(img)

                # Checking if FiveCrop and TenCrop can be printed as string
                transform.__repr__()
                five_crop.__repr__()

                img_clone = np.array(img)
                if should_vflip:
                    vflipped_img = img_clone[::-1]
                    expected_output += five_crop(vflipped_img)
                else:
                    hflipped_img = img_clone[:,::-1]
                    expected_output += five_crop(hflipped_img)

                assert len(results) == 10
                for o, e in zip(results, expected_output):
                    assert (o == e).all()

    def test_resize(self):
        height = random.randint(24, 32) * 2
        width = random.randint(24, 32) * 2
        osize = random.randint(5, 12) * 2

        img = torch.ones(3, height, width)
        result = transforms.Compose([
            transforms.TensorToNumpy(),
            transforms.Resize(osize),
            transforms.ToTensor(),
        ])(img)
        assert osize in result.size()
        if height < width:
            assert result.size(1) <= result.size(2)
        elif width < height:
            assert result.size(1) >= result.size(2)

        result = transforms.Compose([
            transforms.TensorToNumpy(),
            transforms.Resize([osize, osize]),
            transforms.ToTensor(),
        ])(img)
        assert osize in result.size()
        assert result.size(1) == osize
        assert result.size(2) == osize

        oheight = random.randint(5, 12) * 2
        owidth = random.randint(5, 12) * 2
        result = transforms.Compose([
            transforms.TensorToNumpy(),
            transforms.Resize((oheight, owidth)),
            transforms.ToTensor(),
        ])(img)
        assert result.size(1) == oheight
        assert result.size(2) == owidth

        result = transforms.Compose([
            transforms.TensorToNumpy(),
            transforms.Resize([oheight, owidth]),
            transforms.ToTensor(),
        ])(img)
        assert result.size(1) == oheight
        assert result.size(2) == owidth

    def test_random_crop(self):
        height = random.randint(10, 32) * 2
        width = random.randint(10, 32) * 2
        oheight = random.randint(5, (height - 2) / 2) * 2
        owidth = random.randint(5, (width - 2) / 2) * 2
        img = torch.ones(3, height, width)
        result = transforms.Compose([
            transforms.TensorToNumpy(),
            transforms.RandomCrop((oheight, owidth)),
            transforms.ToTensor(),
        ])(img)
        assert result.size(1) == oheight
        assert result.size(2) == owidth

        padding = random.randint(1, 20)
        result = transforms.Compose([
            transforms.TensorToNumpy(),
            transforms.RandomCrop((oheight, owidth), padding=padding),
            transforms.ToTensor(),
        ])(img)
        assert result.size(1) == oheight
        assert result.size(2) == owidth

        result = transforms.Compose([
            transforms.TensorToNumpy(),
            transforms.RandomCrop((height, width)),
            transforms.ToTensor()
        ])(img)
        assert result.size(1) == height
        assert result.size(2) == width
        assert np.allclose(img.numpy(), result.numpy())

        result = transforms.Compose([
            transforms.TensorToNumpy(),
            transforms.RandomCrop((height + 1, width + 1), pad_if_needed=True),
            transforms.ToTensor(),
        ])(img)
        assert result.size(1) == height + 1
        assert result.size(2) == width + 1

    def test_pad(self):
        height = random.randint(10, 32) * 2
        width = random.randint(10, 32) * 2
        img = torch.ones(3, height, width)
        padding = random.randint(1, 20)
        result = transforms.Compose([
            transforms.TensorToNumpy(),
            transforms.Pad(padding),
            transforms.ToTensor(),
        ])(img)
        assert result.size(1) == height + 2 * padding
        assert result.size(2) == width + 2 * padding


    def test_pad_with_tuple_of_pad_values(self):
        height = random.randint(10, 32) * 2
        width = random.randint(10, 32) * 2
        img = transforms.TensorToNumpy()(torch.ones(3, height, width))

        padding = tuple([random.randint(1, 20) for _ in range(2)])
        output = transforms.Pad(padding)(img)
        assert (output.shape[1], output.shape[0]) == (width + padding[0] * 2, height + padding[1] * 2)

        padding = tuple([random.randint(1, 20) for _ in range(4)])
        output = transforms.Pad(padding)(img)
        assert output.shape[1] == width + padding[0] + padding[2]
        assert output.shape[0] == height + padding[1] + padding[3]

        # Checking if Padding can be printed as string
        transforms.Pad(padding).__repr__()