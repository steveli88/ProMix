# For CIFAR-10N
python Train_promix.py --batch_size 256 --noise_type aggre --cosine --dataset cifar10 --num_class 10 --rho_range 0.5,0.5 --tau 0.99 --pretrain_ep 10 --noise_mode cifarn --cluster_prior_epoch 100 --start_expand 450 --cluster_file feature_clusters_cifar10_r50_b384_e1000_c1000.pt
python Train_promix.py --batch_size 256 --noise_type rand1 --cosine --dataset cifar10 --num_class 10 --rho_range 0.5,0.5 --tau 0.99 --pretrain_ep 10 --noise_mode cifarn --cluster_prior_epoch 100 --start_expand 450 --cluster_file feature_clusters_cifar10_r50_b384_e1000_c1000.pt
python Train_promix.py --batch_size 256 --noise_type rand2 --cosine --dataset cifar10 --num_class 10 --rho_range 0.5,0.5 --tau 0.99 --pretrain_ep 10 --noise_mode cifarn --cluster_prior_epoch 100 --start_expand 450 --cluster_file feature_clusters_cifar10_r50_b384_e1000_c1000.pt
python Train_promix.py --batch_size 256 --noise_type rand3 --cosine --dataset cifar10 --num_class 10 --rho_range 0.5,0.5 --tau 0.99 --pretrain_ep 10 --noise_mode cifarn --cluster_prior_epoch 100 --start_expand 450 --cluster_file feature_clusters_cifar10_r50_b384_e1000_c1000.pt
python Train_promix.py --batch_size 256 --noise_type worst --cosine --dataset cifar10 --num_class 10 --rho_range 0.5,0.5 --tau 0.99 --pretrain_ep 10 --noise_mode cifarn --cluster_prior_epoch 100 --start_expand 450 --cluster_file feature_clusters_cifar10_r50_b384_e1000_c1000.pt

# For CIFAR-100N
python Train_promix.py --batch_size 256 --noise_type noisy100 --cosine --dataset cifar100 --num_class 100 --rho_range 0.5,0.5 --tau 0.95 --pretrain_ep 30 --debias_output 0.5 --debias_pl 0.5 --noise_mode cifarn --cluster_prior_epoch 100 --start_expand 450 --cluster_file feature_clusters_cifar100_r50_b384_e1000_c1000.pt
