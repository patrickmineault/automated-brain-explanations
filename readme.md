<h1 align="center">   🧠 Automated brain explanations 🧠</h1>
<p align="center">
<img src="https://img.shields.io/badge/license-mit-blue.svg">
  <img src="https://img.shields.io/badge/python-3.9--3.11-blue">
  <img src="https://img.shields.io/badge/numpy->=2.0-blue">
</p>  

**How does the brain process language?** We've been studying this question using large-scale brain-imaging datasets collected from human subjects as they read and listen to stories.
Along the way, we've used LLMs to help us predict and explain patterns in this data and found a bunch of cool things!
This repo contains code for doing these analyses & applying the tools we've developed to various domains.

### Reference
**This repo contains code underlying two neuroscience studies:**

<details>
<summary>Generative causal testing to bridge data-driven models and scientific theories in language neuroscience (<a href="https://arxiv.org/abs/2410.00812">Antonello*, Singh*, et al., 2024, arXiv</a>)
</summary>
<br>
</details>
<details>
<summary>Evaluating scientific theories as predictive models in language neuroscience (Singh*, Antonello*, et al. 2024, in prep)
</summary>
<br>
</details>
<br>

**This repo also contains code for experiments in three ML papers** (for a simple scikit-learn interface to use these, see [imodelsX](https://github.com/csinva/imodelsX)):
<details>
<summary>Augmenting interpretable models with large language models during training (<a href="https://www.nature.com/articles/s41467-023-43713-1">Singh et al. 2023, Nature communications</a>)
</summary>
<br>
</details>
<details>
<summary>QA-Emb: Crafting interpretable Embeddings by asking LLMs questions (<a href="https://arxiv.org/abs/2405.16714">Benara*, Singh* et al. 2024, NeurIPS</a>)
</summary>
<br>
</details>
<details>
<summary>SASC: Explaining black box text modules in natural language with language models (<a href="https://arxiv.org/abs/2305.09863">Singh*, Hsu*, et al. 2023, NeurIPS workshop</a>)
</summary>
<br>
SASC takes in a text module and produces a natural explanation for it that describes what it types of inputs elicit the largest response from the module (see Fig below). The GCT paper tests this in detail in an fMRI setting.
<br>
<img src="https://microsoft.github.io/automated-brain-explanations/fig.svg?sanitize=True&kill_cache=1" width="90%">

SASC is similar to the nice [concurrent paper](https://github.com/openai/automated-interpretability) by OpenAI, but simplifies explanations to describe the function rather than produce token-level activations. This makes it simpler/faster, and makes it more effective at describing semantic functions from limited data (e.g. fMRI voxels) but worse at finding patterns that depend on sequences / ordering.

To use with <a href="https://github.com/csinva/imodelsX">imodelsX</a>, install with `pip install imodelsx` then the below shows a quickstart example.

```python
from imodelsx import explain_module_sasc
# a toy module that responds to the length of a string
mod = lambda str_list: np.array([len(s) for s in str_list])

# a toy dataset where the longest strings are animals
text_str_list = ["red", "blue", "x", "1", "2", "hippopotamus", "elephant", "rhinoceros"]
explanation_dict = explain_module_sasc(
    text_str_list,
    mod,
    ngrams=1,
)
```
</details>
<br>

### Setting up

**Dataset**
- The `data/decoding` folder contains a quickstart easy example for TR-level decoding
  - it has everything needed, but if you want to visualize the results on a flatmap, you need to download the relevant PCs from [here](https://utexas.box.com/s/7ur0fsr52nephxp96hs5dxm99rk2v1u0)
- to quickstart, just download the responses / wordsequences for 3 subjects from the [encoding scaling laws paper](https://utexas.app.box.com/v/EncodingModelScalingLaws/folder/230420528915)
  - this is all the data you need if you only want to analyze 3 subjects and don't want to make flatmaps
- to run Eng1000, need to grab `em_data` directory from [here](https://github.com/HuthLab/deep-fMRI-dataset) and move its contents to `{root_dir}/em_data`
- for more, download data with `python experiments/00_load_dataset.py`
    - create a `data` dir under wherever you run it and will use [datalad](https://github.com/datalad/datalad) to download the preprocessed data as well as feature spaces needed for fitting [semantic encoding models](https://www.nature.com/articles/nature17637)
- to make flatmaps, need to set [pycortex filestore] to `{root_dir}/ds003020/derivative/pycortex-db/`

**Code**
- `pip install -e .` from the repo directory to locally install the `neuro` package
- set `neuro.config.root_dir/data` to where you put all the data
  - loading responses
    - `neuro.data.response_utils` function `load_response`
    - loads responses from at `{neuro.config.root_dir}/ds003020/derivative/preprocessed_data/{subject}`, where they are stored in an h5 file for each story, e.g. `wheretheressmoke.h5`
  - loading stimulus
    - `ridge_utils.features.stim_utils` function `load_story_wordseqs`
    - loads textgrids from `{root_dir}/ds003020/derivative/TextGrids`, where each story has a TextGrid file, e.g. `wheretheressmoke.TextGrid`
    - uses `{root_dir}/ds003020/derivative/respdict.json` to get the length of each story
- `python experiments/02_fit_encoding.py`
    - This script takes many relevant arguments through argparse

### Reference
- see related [fMRI experiments](https://github.com/csinva/fmri)
- built from [this template](https://github.com/csinva/cookiecutter-ml-research)
