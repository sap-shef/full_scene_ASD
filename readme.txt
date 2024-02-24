inferenceTalkNetDeiT.py (provide model location) -> postprocess results which collates results per track.json to video clip -> python3 format_asd.py asd_results_dir data_set save_path: video clipwise results into video framewise results -> run map.slurm 

python inferTalkNetDeiT.py --evalDataType test
python utils/postprocess_eval.py --saveRes

1_att_head/exp/0.0/model/model_0014.model = 57%
model_0011.model = 49%
/users/acp21jrc/TalkNet_ViT/model_0001.model = 49%
/users/acp21jrc/TalkNet_ViT/model_0005.model = 53%
/users/acp21jrc/TalkNet_ViT/model_0007.model = 57%
/users/acp21jrc/TalkNet_ViT/model_0010.model = 64%
/users/acp21jrc/TalkNet_ViT/model_0011.model = 

to determine the disfluency between the track.json files and the background_speaker.json files we must first find where each file originates from.

Regenrate bboxes_per_track to include the candidate speaker -> rerun tensorgrabber.get_background_speakers_bboxes an ensure that the candidate speaker is elimanated in EVERY frame in ALL bboxes_per_track files -> if this does not perform as expected, use pid to eliminate candidate speaker bounding box from bboxes_per_track instead of doing so via bbox coordinates.

Rewrite script to extract .json files for candidate speaker and .json files for all background speakers (including candidate speaker) for all folds:
	- must be compatible with oracle tracking results (using vid_num as file name as oppose to vid_id)
	- must create the json files for the candidate speaker tracks
	- must create the all_bboxes_per_track.json INCLUDING the candidate speaker include pid 
	- must replace preprocessing for train and val fold
	- must replace process_tracking_result.py for test fold
	- if statement to change the base directory
	

to run:
	- ego4d/csv/{active_speaker_train.csv, active_speaker_val.csv}
	- ego4d/bbox/
	- infer/csv/{active_speaker_test.csv}
	- infer/bbox
	- tensors
	
starting without processed ASD annotation:
	- python annot_preprocess.py --basePath {path to Ego4d_TalkNet_ASD} --split train
	- python annot_preprocess.py --basePath {path to Ego4d_TalkNet_ASD} --split val
	- python annot_preprocess.py --basePath {path to Ego4d_TalkNet_ASD} --split
	
starting with processed ASD annotations:
	# Run the following commands across multiple HPC sessions simultaneously:
		# train/val: 
		#   python tensor_grabber.py --annotPath {path to ego4d/csv & bbox} --split {train/val} --dataPath {path to video_imgs} --savePath {path to save tensors}
		# testing ASD on Ego4D reconfigured validation fold:
		#   python tensor_grabber.py --annotPath {path to infer/csv & bbox} --split val --dataPath {path to video_imgs} --savePath {path to save tensors}
	# Run across single session to fill any missing tracks:
		# train/val:
		#   python tensor_grabber.py --annotPath {path to ego4d/csv & bbox} --split {train/val} --dataPath {path to video_imgs} --savePath {path to save tensors} --fillPass
		# testing ASD on Ego4D reconfigured validation fold:
		#   python tensor_grabber.py --annotPath {path to infer/csv & bbox} --split val --dataPath {path to video_imgs} --savePath {path to save tensors} --fillPass

train TalkNet+DeiT:
	python trainTalkNetDeiT.py
	
infer TalkNet+DeiT:
	python inferTalkNetDeiT.py
	
post process:
	postprocess_eval.py 
