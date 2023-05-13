# RWKV-Inference-Experiments

This repo contains all the code used to collect performance information about different models.

## Models used
 - [ ] RWKV-4
 - [ ] Pythia
 - [ ] GPT-Neo

 Should we also test with other models? Like Hyenna or SpikeGPT?

## Methodology

To evaluate the inference performance of each model we could generate 1000 tokens for some prompt.
One prompt we could use is the `dragon prompt`:

```text
\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese.
```

Another prompt we could use if an extract from the Project Gutenberg.