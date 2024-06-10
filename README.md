# HESC

This repository contains the code and the data for the paper "Being Human Supporters: Guiding LLMs for 
Emotional Support Conversation via Chain-of-Thought"

## Data

The original ESConv dataset is available under `esconv/` directory. You can run the `process_esconv.sh` to
convert the data into a format that we use in our experiments. It will create a json file inside the same folder
called `conversations.json`. You can run the script with the following command:

```sh
bash process_esconv.sh
```


## Experiments

For our experiments we use LLaMa v2 chat models with 4bit quantization. You can follow the instruction in the following
links to get access to [LLaMa2-7b-chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) models on huggingface.

All of the experiments are conducted using the `transformers` library. We use bitsandbytes to quantize the models.

You can run the experiments in the paper using the following commands:

```sh
cd prompting
bash llama7b.sh
```


Then you can postprocess the generated responses using `prompting/postprocess.py` script. A sample of the generated data
is available in `data/` directory. Each file contains one incomplete conversation and a few continuations using 
different strategies. Along with this information we also provide the exact prompt that we used to generate each continuation.