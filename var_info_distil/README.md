# Variational Information Distillation for Knowledge Transfer

This is an official PyTorch implementation of "Variational Information Distillation for Knowledge Transfer" published as a conference paper at CVPR 2019 (http://openaccess.thecvf.com/content_CVPR_2019/papers/Ahn_Variational_Information_Distillation_for_Knowledge_Transfer_CVPR_2019_paper.pdf).

Please cite our work if you find it useful:
```
@inproceedings{ahn2019variational,
  title={Variational information distillation for knowledge transfer},
  author={Ahn, Sungsoo and Hu, Shell Xu and Damianou, Andreas and Lawrence, Neil D and Dai, Zhenwen},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={9163--9171},
  year={2019}
}
```


### Dependencies
This code was implemented based on pytorch 1.3.1 (https://github.com/pytorch/pytorch) with torch 0.4.2 and pytorch-ignite 0.3.0 (https://github.com/pytorch/ignite) libraries. You can install it using the requirements.txt file.

```
pip install -r requirements.txt
```

### Using the implementation
In what follows, we provide the description for using the implementation for training a student model (wide residual network with depth 16 and width 1) with knowledge transfer from the teacher model (wide residual network with depth 40 and width 2) as a classifier for the CIFAR-10 dataset. You can vary the architecture of the teacher and the student model with appropriate choice of arguments.

First, train the teacher model (wide residual network with depth 40 and width 2) to be used for guiding the student model. 

```
python train_without_transfer.py \
--depth 40 \
--width 2 \
--state-dict-path ./state_dict/teacher.th \
--device $DEVICE \
--data-dir $DATA_DIR \
--num-workers $NUM_WORKERS
```

Next, train the student model (wide residual network with depth 16 and width 1) using variational information distillation.
```
python train_with_transfer.py \
--student-depth 16 \
--student-width 1 \
--teacher-depth 40 \
--teacher-width 2 \
--variational-information-distillation-factor 0.1 \
--knowledge-distillation-factor 1.0 \
--knowledge-distillation-temperature 2.0 \
--state-dict-path ./state_dict/student_with_transfer.th \
--teacher-state-dict-path ./state_dict/teacher.th \
--device $DEVICE \
--data-dir $DATA_DIR \
--num-workers $NUM_WORKERS
```

One can also train the student model (wide residual network with depth 16 and width 1) without transfer learning to validate the improvement from transfer learning. 
```
python train_without_transfer.py \
--depth 16 \
--width 1 \
--state-dict-path ./state_dict/student.th \
--device $DEVICE \
--data-dir $DATA_DIR \
--num-workers $NUM_WORKERS
```

### Authors of the code 
Sungsoo Ahn, Shell Xu Hu  
