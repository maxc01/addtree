.PHONY: run_addtree
run_addtree:
	SINGULARITYENV_PYTHONPATH=/esat/vauxite/xma/sing_images/images_def/projects/addtree SINGULARITYENV_CUDA_VISIBLE_DEVICES=0 singularity exec --nv /esat/vauxite/xma/sing_images/images_def/working_images/automl.sif python addtree_model_compression_vgg16.py --prune_epochs 3 --checkpoints_dir ./checkpoints_addtree --pretrained ./cifar10_vgg16_best.pt

.PHONY: run_smac
run_smac:
	SINGULARITYENV_CUDA_VISIBLE_DEVICES=0 singularity exec --nv /esat/vauxite/xma/sing_images/images_def/working_images/automl.sif python smac_model_compression_vgg16.py --prune_epochs 3 --checkpoints_dir ./checkpoints_smac --pretrained ./cifar10_vgg16_best.pt

.PHONY: run_random
run_random:
	SINGULARITYENV_CUDA_VISIBLE_DEVICES=0 singularity exec --nv /esat/vauxite/xma/sing_images/images_def/working_images/automl.sif python random_model_compression_vgg16.py --prune_epochs 3 --checkpoints_dir ./checkpoints_random --pretrained ./cifar10_vgg16_best.pt

.PHONY: clean
clean:
	rm -rf ./checkpoints_random ./checkpoints_addtree ./checkpoints_smac ./smac3-output_2020*


.PHONY: info
info:
	bash ./script/stats.sh


.PHONY: collect_multiple
collect_multiple:
	@bash ./script/collect_multiple.sh

.PHONY: collect_single
collect_single:
	@bash ./script/collect_single.sh
