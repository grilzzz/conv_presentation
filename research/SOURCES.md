# Карта утверждений и источников

| Утверждение | Основной источник |
|---|---|
| Dilated convolution расширяет receptive field без потери разрешения | Yu & Koltun, 2015: https://arxiv.org/abs/1511.07122 |
| Одинаковые повторяющиеся dilation rates могут вызывать gridding | Wang et al., 2017: https://arxiv.org/abs/1702.08502 |
| Depthwise separable convolution состоит из depthwise и pointwise стадий и снижает вычисления | MobileNet: https://arxiv.org/abs/1704.04861 |
| Факторизация разделяет пространственные и межканальные зависимости | Xception: https://openaccess.thecvf.com/content_cvpr_2017/html/Chollet_Xception_Deep_Learning_CVPR_2017_paper.html |
| Deformable convolution обучает offsets без отдельной разметки | Dai et al., 2017: https://arxiv.org/abs/1703.06211 |
| DCNv2 добавляет modulation mask | Zhu et al., 2019: https://openaccess.thecvf.com/content_CVPR_2019/html/Zhu_Deformable_ConvNets_V2_More_Deformable_Better_Results_CVPR_2019_paper.html |
| CIFAR‑10: 60 000 изображений 32×32, 10 классов, split 50k/10k | University of Toronto: https://www.cs.toronto.edu/~kriz/cifar.html |
| `groups=in_channels` реализует depthwise convolution в PyTorch | PyTorch Conv2d docs: https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html |
| Формат offset tensor для DeformConv2d | Torchvision docs: https://docs.pytorch.org/vision/stable/generated/torchvision.ops.DeformConv2d.html |
| Все численные результаты проекта | `outputs_full/metrics.csv`, `outputs_full/training_history.csv` |
