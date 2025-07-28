NAME="train"
python3 combine.py \
--first 50000 \
--o ../data_s/nvars=10-nvals=7-zseed=42-seq_len=10-nbranches=3-xseed=42-mean=0.0-std=1.0-sseed=42-N=100000/${NAME}.bin \
--i ../data_s/nvars=10-nvals=7-zseed=42-seq_len=10-nbranches=3-xseed=42-mean=0.0-std=1.0-sseed=42-N=100000/${NAME}.0.bin \
../data_s/nvars=10-nvals=7-zseed=42-seq_len=10-nbranches=3-xseed=42-mean=0.0-std=1.0-sseed=42-N=100000/${NAME}.1.bin \
../data_s/nvars=10-nvals=7-zseed=42-seq_len=10-nbranches=3-xseed=42-mean=0.0-std=1.0-sseed=42-N=100000/${NAME}.2.bin \
../data_s/nvars=10-nvals=7-zseed=42-seq_len=10-nbranches=3-xseed=42-mean=0.0-std=1.0-sseed=42-N=100000/${NAME}.3.bin \
../data_s/nvars=10-nvals=7-zseed=42-seq_len=10-nbranches=3-xseed=42-mean=0.0-std=1.0-sseed=42-N=100000/${NAME}.4.bin \
../data_s/nvars=10-nvals=7-zseed=42-seq_len=10-nbranches=3-xseed=42-mean=0.0-std=1.0-sseed=42-N=100000/${NAME}.5.bin \
../data_s/nvars=10-nvals=7-zseed=42-seq_len=10-nbranches=3-xseed=42-mean=0.0-std=1.0-sseed=42-N=100000/${NAME}.6.bin \
../data_s/nvars=10-nvals=7-zseed=42-seq_len=10-nbranches=3-xseed=42-mean=0.0-std=1.0-sseed=42-N=100000/${NAME}.7.bin \
../data_s/nvars=10-nvals=7-zseed=42-seq_len=10-nbranches=3-xseed=42-mean=0.0-std=1.0-sseed=42-N=100000/${NAME}.8.bin \
../data_s/nvars=10-nvals=7-zseed=42-seq_len=10-nbranches=3-xseed=42-mean=0.0-std=1.0-sseed=42-N=100000/${NAME}.9.bin \
../data_s/nvars=10-nvals=7-zseed=42-seq_len=10-nbranches=3-xseed=42-mean=0.0-std=1.0-sseed=42-N=100000/${NAME}.10.bin \
../data_s/nvars=10-nvals=7-zseed=42-seq_len=10-nbranches=3-xseed=42-mean=0.0-std=1.0-sseed=42-N=100000/${NAME}.11.bin \
../data_s/nvars=10-nvals=7-zseed=42-seq_len=10-nbranches=3-xseed=42-mean=0.0-std=1.0-sseed=42-N=100000/${NAME}.12.bin \
../data_s/nvars=10-nvals=7-zseed=42-seq_len=10-nbranches=3-xseed=42-mean=0.0-std=1.0-sseed=42-N=100000/${NAME}.13.bin \
../data_s/nvars=10-nvals=7-zseed=42-seq_len=10-nbranches=3-xseed=42-mean=0.0-std=1.0-sseed=42-N=100000/${NAME}.14.bin \
../data_s/nvars=10-nvals=7-zseed=42-seq_len=10-nbranches=3-xseed=42-mean=0.0-std=1.0-sseed=42-N=100000/${NAME}.15.bin \
../data_s/nvars=10-nvals=7-zseed=42-seq_len=10-nbranches=3-xseed=42-mean=0.0-std=1.0-sseed=42-N=100000/${NAME}.16.bin \
../data_s/nvars=10-nvals=7-zseed=42-seq_len=10-nbranches=3-xseed=42-mean=0.0-std=1.0-sseed=42-N=100000/${NAME}.17.bin \
../data_s/nvars=10-nvals=7-zseed=42-seq_len=10-nbranches=3-xseed=42-mean=0.0-std=1.0-sseed=42-N=100000/${NAME}.18.bin \
../data_s/nvars=10-nvals=7-zseed=42-seq_len=10-nbranches=3-xseed=42-mean=0.0-std=1.0-sseed=42-N=100000/${NAME}.19.bin \


NAME="test"
python3 combine.py \
--o ../data_s/nvars=10-nvals=7-zseed=42-seq_len=10-nbranches=3-xseed=42-mean=0.0-std=1.0-sseed=42-N=100000/${NAME}.bin \
--i ../data_s/nvars=10-nvals=7-zseed=42-seq_len=10-nbranches=3-xseed=42-mean=0.0-std=1.0-sseed=42-N=100000/${NAME}.0.bin \
../data_s/nvars=10-nvals=7-zseed=42-seq_len=10-nbranches=3-xseed=42-mean=0.0-std=1.0-sseed=42-N=100000/${NAME}.1.bin \
../data_s/nvars=10-nvals=7-zseed=42-seq_len=10-nbranches=3-xseed=42-mean=0.0-std=1.0-sseed=42-N=100000/${NAME}.2.bin \
../data_s/nvars=10-nvals=7-zseed=42-seq_len=10-nbranches=3-xseed=42-mean=0.0-std=1.0-sseed=42-N=100000/${NAME}.3.bin \
../data_s/nvars=10-nvals=7-zseed=42-seq_len=10-nbranches=3-xseed=42-mean=0.0-std=1.0-sseed=42-N=100000/${NAME}.4.bin \
../data_s/nvars=10-nvals=7-zseed=42-seq_len=10-nbranches=3-xseed=42-mean=0.0-std=1.0-sseed=42-N=100000/${NAME}.5.bin \
../data_s/nvars=10-nvals=7-zseed=42-seq_len=10-nbranches=3-xseed=42-mean=0.0-std=1.0-sseed=42-N=100000/${NAME}.6.bin \
../data_s/nvars=10-nvals=7-zseed=42-seq_len=10-nbranches=3-xseed=42-mean=0.0-std=1.0-sseed=42-N=100000/${NAME}.7.bin \
../data_s/nvars=10-nvals=7-zseed=42-seq_len=10-nbranches=3-xseed=42-mean=0.0-std=1.0-sseed=42-N=100000/${NAME}.8.bin \
../data_s/nvars=10-nvals=7-zseed=42-seq_len=10-nbranches=3-xseed=42-mean=0.0-std=1.0-sseed=42-N=100000/${NAME}.9.bin \
../data_s/nvars=10-nvals=7-zseed=42-seq_len=10-nbranches=3-xseed=42-mean=0.0-std=1.0-sseed=42-N=100000/${NAME}.10.bin \
../data_s/nvars=10-nvals=7-zseed=42-seq_len=10-nbranches=3-xseed=42-mean=0.0-std=1.0-sseed=42-N=100000/${NAME}.11.bin \
../data_s/nvars=10-nvals=7-zseed=42-seq_len=10-nbranches=3-xseed=42-mean=0.0-std=1.0-sseed=42-N=100000/${NAME}.12.bin \
../data_s/nvars=10-nvals=7-zseed=42-seq_len=10-nbranches=3-xseed=42-mean=0.0-std=1.0-sseed=42-N=100000/${NAME}.13.bin \
../data_s/nvars=10-nvals=7-zseed=42-seq_len=10-nbranches=3-xseed=42-mean=0.0-std=1.0-sseed=42-N=100000/${NAME}.14.bin \
../data_s/nvars=10-nvals=7-zseed=42-seq_len=10-nbranches=3-xseed=42-mean=0.0-std=1.0-sseed=42-N=100000/${NAME}.15.bin \
../data_s/nvars=10-nvals=7-zseed=42-seq_len=10-nbranches=3-xseed=42-mean=0.0-std=1.0-sseed=42-N=100000/${NAME}.16.bin \
../data_s/nvars=10-nvals=7-zseed=42-seq_len=10-nbranches=3-xseed=42-mean=0.0-std=1.0-sseed=42-N=100000/${NAME}.17.bin \
../data_s/nvars=10-nvals=7-zseed=42-seq_len=10-nbranches=3-xseed=42-mean=0.0-std=1.0-sseed=42-N=100000/${NAME}.18.bin \
../data_s/nvars=10-nvals=7-zseed=42-seq_len=10-nbranches=3-xseed=42-mean=0.0-std=1.0-sseed=42-N=100000/${NAME}.19.bin \

NAME="dev"
python3 combine.py \
--o ../data_s/nvars=10-nvals=7-zseed=42-seq_len=10-nbranches=3-xseed=42-mean=0.0-std=1.0-sseed=42-N=100000/${NAME}.bin \
--i ../data_s/nvars=10-nvals=7-zseed=42-seq_len=10-nbranches=3-xseed=42-mean=0.0-std=1.0-sseed=42-N=100000/${NAME}.0.bin \
../data_s/nvars=10-nvals=7-zseed=42-seq_len=10-nbranches=3-xseed=42-mean=0.0-std=1.0-sseed=42-N=100000/${NAME}.1.bin \
../data_s/nvars=10-nvals=7-zseed=42-seq_len=10-nbranches=3-xseed=42-mean=0.0-std=1.0-sseed=42-N=100000/${NAME}.2.bin \
../data_s/nvars=10-nvals=7-zseed=42-seq_len=10-nbranches=3-xseed=42-mean=0.0-std=1.0-sseed=42-N=100000/${NAME}.3.bin \
../data_s/nvars=10-nvals=7-zseed=42-seq_len=10-nbranches=3-xseed=42-mean=0.0-std=1.0-sseed=42-N=100000/${NAME}.4.bin \
../data_s/nvars=10-nvals=7-zseed=42-seq_len=10-nbranches=3-xseed=42-mean=0.0-std=1.0-sseed=42-N=100000/${NAME}.5.bin \
../data_s/nvars=10-nvals=7-zseed=42-seq_len=10-nbranches=3-xseed=42-mean=0.0-std=1.0-sseed=42-N=100000/${NAME}.6.bin \
../data_s/nvars=10-nvals=7-zseed=42-seq_len=10-nbranches=3-xseed=42-mean=0.0-std=1.0-sseed=42-N=100000/${NAME}.7.bin \
../data_s/nvars=10-nvals=7-zseed=42-seq_len=10-nbranches=3-xseed=42-mean=0.0-std=1.0-sseed=42-N=100000/${NAME}.8.bin \
../data_s/nvars=10-nvals=7-zseed=42-seq_len=10-nbranches=3-xseed=42-mean=0.0-std=1.0-sseed=42-N=100000/${NAME}.9.bin \
../data_s/nvars=10-nvals=7-zseed=42-seq_len=10-nbranches=3-xseed=42-mean=0.0-std=1.0-sseed=42-N=100000/${NAME}.10.bin \
../data_s/nvars=10-nvals=7-zseed=42-seq_len=10-nbranches=3-xseed=42-mean=0.0-std=1.0-sseed=42-N=100000/${NAME}.11.bin \
../data_s/nvars=10-nvals=7-zseed=42-seq_len=10-nbranches=3-xseed=42-mean=0.0-std=1.0-sseed=42-N=100000/${NAME}.12.bin \
../data_s/nvars=10-nvals=7-zseed=42-seq_len=10-nbranches=3-xseed=42-mean=0.0-std=1.0-sseed=42-N=100000/${NAME}.13.bin \
../data_s/nvars=10-nvals=7-zseed=42-seq_len=10-nbranches=3-xseed=42-mean=0.0-std=1.0-sseed=42-N=100000/${NAME}.14.bin \
../data_s/nvars=10-nvals=7-zseed=42-seq_len=10-nbranches=3-xseed=42-mean=0.0-std=1.0-sseed=42-N=100000/${NAME}.15.bin \
../data_s/nvars=10-nvals=7-zseed=42-seq_len=10-nbranches=3-xseed=42-mean=0.0-std=1.0-sseed=42-N=100000/${NAME}.16.bin \
../data_s/nvars=10-nvals=7-zseed=42-seq_len=10-nbranches=3-xseed=42-mean=0.0-std=1.0-sseed=42-N=100000/${NAME}.17.bin \
../data_s/nvars=10-nvals=7-zseed=42-seq_len=10-nbranches=3-xseed=42-mean=0.0-std=1.0-sseed=42-N=100000/${NAME}.18.bin \
../data_s/nvars=10-nvals=7-zseed=42-seq_len=10-nbranches=3-xseed=42-mean=0.0-std=1.0-sseed=42-N=100000/${NAME}.19.bin \