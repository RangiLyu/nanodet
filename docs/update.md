# Update Notes

* [2021.08.28] Refactor data processing pipeline and support multi-scale training (#311).

* [2021.05.30] Release ncnn int8 models, and new pre-trained models with ShuffleNetV2-1.5x backbone. Much higher mAP but still realtime(**26.8mAP 21.53ms**).

* [2021.03.12] Apply the **Transformer** encoder to NanoDet! Introducing **NanoDet-t**, which replaces the PAN in NanoDet-m with a **TAN(Transformer Attention Net)**,  gets 21.7 mAP(+1.1) on COCO val 2017. Check [nanodet-t.yml](config/Transformer/nanodet-t.yml) for more details.

* [2021.03.03] Update **Nanodet-m-416** COCO pretrained model. **COCO mAP(0.5:0.95)=23.5**. Download in [Model Zoo](#model-zoo).

* [2021.02.03] Support [EfficientNet-Lite](https://github.com/RangiLyu/EfficientNet-Lite) and [Rep-VGG](https://github.com/DingXiaoH/RepVGG) backbone. Please check the [config folder](config/). Download models in [Model Zoo](#model-zoo)

* [2021.01.10] **NanoDet-g** with lower memory access cost, which designed for edge NPU or GPU, is now available!
  Check [config/nanodet-g.yml](config/nanodet-g.yml) and download in [Model Zoo](#model-zoo).

* [2020.12.19] [MNN python and cpp demos](demo_mnn/) are available.

* [2020.12.05] Support voc .xml format dataset! Refer to [config/nanodet_custom_xml_dataset.yml](config/nanodet_custom_xml_dataset.yml).

* [2020.12.01] Great thanks to nihui, now you can try NanoDet running in web browser! 👉 https://nihui.github.io/ncnn-webassembly-nanodet/
