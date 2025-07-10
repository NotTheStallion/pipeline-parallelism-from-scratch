gpipe:
	torchrun --nproc-per-node 4 gpipe.py

1f1b:
	torchrun --nproc-per-node 4 1f1b.py

test_gpipe:
	torchrun --nproc-per-node 4 test_gpipe.py

profile:
	nsys profile --sample process-tree --stats true --trace cuda,osrt --force-overwrite true --output profile.nsys-rep torchrun --nproc-per-node 4 gpipe.py



# nsys-ui profile.nsys-rep