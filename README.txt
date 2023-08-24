
##@20230824

##QA
torchrun --nproc_per_node 2 demo_chat_GEN.py --ckpt_dir llama-2-13b-chat/ --tokenizer_path tokenizer.model --max_seq_len 2048 --max_batch_size 16

torchrun --nproc_per_node 2 demo_chat_GEN.py --ckpt_dir llama-2-13b-chat/ --tokenizer_path tokenizer.model --max_seq_len 2048 --max_batch_size 16 1>zz.13B.QA.std.o 2>zz.std.e

##Chat with PDF
torchrun --nproc_per_node 2 demo_langchain_QA.py --ckpt_dir llama-2-13b-chat/ --tokenizer_path tokenizer.model --max_seq_len 2048 --max_batch_size 16

>>TXT
torchrun --nproc_per_node 2 demo_langchain_QA.py --ckpt_dir llama-2-13b-chat/ --tokenizer_path tokenizer.model --max_seq_len 2048 --max_batch_size 16 1>zz.13B.DOCQA_TXT.std.o 2>zz.std.e

>>PDF
torchrun --nproc_per_node 2 demo_langchain_QA.py --ckpt_dir llama-2-13b-chat/ --tokenizer_path tokenizer.model --max_seq_len 2048 --max_batch_size 16 1>zz.13B.DOCQA_PDF.std.o 2>zz.std.e

##
torchrun --nproc_per_node 2 demo_chat_QE.py --ckpt_dir llama-2-13b-chat/ --tokenizer_path tokenizer.model --max_seq_len 2048 --max_batch_size 16
