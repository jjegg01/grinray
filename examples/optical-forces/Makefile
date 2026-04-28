# Makefile for generating all the force data used in https://arxiv.org/abs/2604.14917

.PHONY: force-scans grin grin-sphere grin-cap grin-cap-neg

force-scans: output-forces/hemisphere.csv output-forces/cap.csv output-forces/cone.csv output-forces/cornet.csv grin

output-forces/hemisphere.csv:
	cargo run --release -- --shape hemisphere scan $@

output-forces/cap.csv:
	cargo run --release -- --shape hemisphere-shell scan $@

output-forces/cone.csv:
	cargo run --release -- --shape cone scan $@

output-forces/cornet.csv:
	cargo run --release -- --shape cone-shell scan $@

grin: grin-sphere grin-cap-parallel grin-cap-antiparallel

grin-sphere:
	./utils/run-grin-sphere output-forces/grin-sphere/
	
grin-cap-parallel:
	./utils/run-grin-cap output-forces/grin-cap-parallel/

grin-cap-antiparallel:
	./utils/run-grin-cap --max-strength -0.01 output-forces/grin-cap-antiparallel/

clean:
	rm -f output-forces/{hemisphere,cap,cone,cornet}.csv
	rm -rf output-forces/grin-sphere
	rm -rf output-forces/grin-cap-parallel
	rm -rf output-forces/grin-cap-antiparallel