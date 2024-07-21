ckt_path="weights/pretrained_all.pth"
gpu_id=0

# demo: do zero-shot anomaly detection for a single image
CUDA_VISIBLE_DEVICES=$gpu_id python test.py --testing_model image --ckt_path $ckt_path --save_fig True \
 --image_path asset/img.png --class_name candle --save_name test.png