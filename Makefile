gpipe:
	torchrun --nproc-per-node 4 gpipe.py

1f1b:
	torchrun --nproc-per-node 4 1f1b.py

tspipe:
	torchrun --nproc-per-node 4 standard-kd.py

zbts:
	torchrun --nproc-per-node 4 zb.py

test_gpipe:
	torchrun --nproc-per-node 4 test_gpipe.py

profile:
	nsys profile --sample process-tree --stats true --trace cuda,osrt --force-overwrite true --output profile.nsys-rep torchrun --nproc-per-node 4 gpipe.py

prof:
	nsys profile --sample process-tree --stats false --trace nvtx --force-overwrite true --output profile.nsys-rep torchrun --nproc-per-node 4 1f1b.py

prof_ts:
	nsys profile --sample process-tree --stats false --trace nvtx --force-overwrite true --output profile.nsys-rep torchrun --nproc-per-node 4 standard-kd.py

prof_zb:
	nsys profile --sample process-tree --stats false --trace nvtx --force-overwrite true --output profile.nsys-rep torchrun --nproc-per-node 4 zb.py

.PHONY: prof

ui:
	nsys-ui profile.nsys-rep

clean:
	rm *.png