<h1 align="center">   <img src="https://microsoft.github.io/augmented-interpretable-models/auggam_gif.gif" width="25%"> Automated brain explanations <img src="https://microsoft.github.io/augmented-interpretable-models/auggam_gif.gif" width="25%"></h1>


This repo contains the code underlying two neuroscience studies:

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

and two ML papers (For a simple scikit-learn interface to use these, see the [imodelsX library](https://github.com/csinva/imodelsX)):
<details>
<summary>QA-Emb: Crafting Interpretable Embeddings by Asking LLMs Questions (<a href="https://arxiv.org/abs/2405.16714">Benara*, Singh* et al. 2024, NeurIPS</a>)
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

# Setting up

**Dataset**
- The `data/decoding` folder contains a quickstart easy example for TR-level decoding
- to quickstart, just download the responses / wordsequences for 3 subjects from the [encoding scaling laws paper](https://utexas.app.box.com/v/EncodingModelScalingLaws/folder/230420528915)
  - this is all the data you need if you only want to analyze 3 subjects and don't want to make flatmaps
- to run Eng1000, need to grab `em_data` directory from [here](https://github.com/HuthLab/deep-fMRI-dataset) and move its contents to `{root_dir}/em_data`
- for more, download data with `python experiments/00_load_dataset.py`
    - create a `data` dir under wherever you run it and will use [datalad](https://github.com/datalad/datalad) to download the preprocessed data as well as feature spaces needed for fitting [semantic encoding models](https://www.nature.com/articles/nature17637)
- to make flatmaps, need to set [pycortex filestore] to `{root_dir}/ds003020/derivative/pycortex-db/`

**Code**
- `pip install ridge_utils` (for full control, could alternatively `pip install -e ridge_utils_frozen` from the repo directory)
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

# Reference
- See related [fMRI experiments](https://github.com/csinva/fmri)
- Built from [this template](https://github.com/csinva/cookiecutter-ml-research)
