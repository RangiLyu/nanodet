python tools/export_onnx.py --cfg_path  ./config/nanodet-plus-m_320_mobileone.yml   --model_path   /home/notebook/code/personal/model_compression/nanodet-main/workspace/nanodet-plus-m_320/model_best/model_best.ckpt 
python3 -m onnxsim nanodet.onnx mobileone.onnx
cp mobileone.onnx  ../../privacy/convertor/ncnn/build/tools/onnx/
cd ../../privacy/convertor/ncnn/build/tools/onnx/




##./im_detection_0616_biaoti_dialog_neiwaixiao_0620_single2



### ./im_detection_0616_biaoti_dialog_neiwaixiao_0620_single1_nodialog/




###nanodet-plus-m-1.5x_416_0621_tmp
#nanodet-plus-m-1.5x_416_0621
#im_detection_0615_biaoti_dialog_neiwaixiao
