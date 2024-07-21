# pre-trained from MVTec and ColonDB
ckt_path="weights/pretrained_mvtec_colondb.pth"
gpu_id=0

CUDA_VISIBLE_DEVICES=$gpu_id python test.py --testing_model dataset --ckt_path $ckt_path --save_fig True --testing_data br35h
CUDA_VISIBLE_DEVICES=$gpu_id python test.py --testing_model dataset --ckt_path $ckt_path --save_fig True --testing_data brain_mri
CUDA_VISIBLE_DEVICES=$gpu_id python test.py --testing_model dataset --ckt_path $ckt_path --save_fig True --testing_data btad
CUDA_VISIBLE_DEVICES=$gpu_id python test.py --testing_model dataset --ckt_path $ckt_path --save_fig True --testing_data clinicdb
CUDA_VISIBLE_DEVICES=$gpu_id python test.py --testing_model dataset --ckt_path $ckt_path --save_fig True --testing_data dagm
CUDA_VISIBLE_DEVICES=$gpu_id python test.py --testing_model dataset --ckt_path $ckt_path --save_fig True --testing_data dtd
CUDA_VISIBLE_DEVICES=$gpu_id python test.py --testing_model dataset --ckt_path $ckt_path --save_fig True --testing_data headct
CUDA_VISIBLE_DEVICES=$gpu_id python test.py --testing_model dataset --ckt_path $ckt_path --save_fig True --testing_data isic
CUDA_VISIBLE_DEVICES=$gpu_id python test.py --testing_model dataset --ckt_path $ckt_path --save_fig True --testing_data mpdd
CUDA_VISIBLE_DEVICES=$gpu_id python test.py --testing_model dataset --ckt_path $ckt_path --save_fig True --testing_data sdd
CUDA_VISIBLE_DEVICES=$gpu_id python test.py --testing_model dataset --ckt_path $ckt_path --save_fig True --testing_data tn3k
CUDA_VISIBLE_DEVICES=$gpu_id python test.py --testing_model dataset --ckt_path $ckt_path --save_fig True --testing_data visa

# pre-trained from Visa and Clinicdb
ckt_path="weights/pretrained_visa_clinicdb.pth"
gpu_id=0

CUDA_VISIBLE_DEVICES=$gpu_id python test.py --testing_model dataset --ckt_path $ckt_path --save_fig True --testing_data colondb
CUDA_VISIBLE_DEVICES=$gpu_id python test.py --testing_model dataset --ckt_path $ckt_path --save_fig True --testing_data mvtec



