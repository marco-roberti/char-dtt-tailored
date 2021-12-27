# Character-Based Data-to-Text Generation
Codebase for the paper ["Copy mechanism and tailored training for character-based data-to-text generation" (Roberti et al., ECML-PKDD 2019)](https://arxiv.org/abs/1904.11838).


## Step-by-step guide
### Requirements
A working Python 3 environment is needed. Required libraries are listed in the `requirements.txt` file, use **one of the following commands** to install them, depending on your environment:
```bash
pip install requirements.txt
# XOR
conda install --file requirements.txt
```

### Training
The `main.py` file is used to train an `EDA_CS`, `EDA_C` or `EDA` model on the dataset on your choice:
```bash
python3 main.py --dataset <dataset> --model <model>
```
The default configuration trains `EDA_CS` on the E2E+ dataset.
Available models are `['e2e+', 'e2e', 'hotel', 'restaurant']`; available datasets are `['eda_cs', 'eda_c', 'eda']`.

Different hyperparameters can be set via argparse (run `python3 main.py -h` for more details).

At the end of the training phase, one checkpoint for each epoch will be stored in the `trained_nets/<timestamp>/` folder, where `timestamp` is the UNIX time of starting the script.

### Generation
The `create_eval_files.py` script will generate both outputs and references files, which can be directly used as inputs for the evaluation script. For example, you can generate on the E2E development set using `ED+ACS` as follows:
```bash
PYTHONPATH=. python3 utils/create_eval_files.py trained_nets/<timestamp>/<checkpoint> --subset dev
```
This will create the `trained_nets/<timestamp>/<checkpoint>.dev.output` and `trained_nets/<timestamp>/<checkpoint>.dev.references` files.

The default configuration uses your `EDA_CS` checkpoint to generate from the E2E+ test dataset's inputs. You can choose a different dataset/subset/architecture via argparse.

### Evaluation
We took advantage of the [E2E NLG Challenge Evaluation metrics](https://github.com/tuetschek/e2e-metrics). Please refer to their repository for detailed instructions.

## Citations
Please use the following BibTeX snippet to cite our work:

```BibTeX
@inproceedings{Roberti2019,
  author    = {Marco Roberti and
               Giovanni Bonetta and
               Rossella Cancelliere and
               Patrick Gallinari},
  title     = {Copy Mechanism and Tailored Training for Character-Based Data-to-Text
               Generation},
  booktitle = {Machine Learning and Knowledge Discovery in Databases - European Conference,
               {ECML} {PKDD} 2019, W{\"{u}}rzburg, Germany, September 16-20,
               2019, Proceedings, Part {II}},
  pages     = {648--664},
  year      = {2019},
  crossref  = {ECMLPKDD2019-2},
  url       = {https://doi.org/10.1007/978-3-030-46147-8\_39},
  doi       = {10.1007/978-3-030-46147-8\_39},
  timestamp = {Mon, 15 Jun 2020 17:05:23 +0200},
  biburl    = {https://dblp.org/rec/conf/pkdd/RobertiBCG19.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
@proceedings{ECMLPKDD2019-2,
  editor    = {Ulf Brefeld and
               {\'{E}}lisa Fromont and
               Andreas Hotho and
               Arno J. Knobbe and
               Marloes H. Maathuis and
               C{\'{e}}line Robardet},
  title     = {Machine Learning and Knowledge Discovery in Databases - European Conference,
               {ECML} {PKDD} 2019, W{\"{u}}rzburg, Germany, September 16-20,
               2019, Proceedings, Part {II}},
  series    = {Lecture Notes in Computer Science},
  volume    = {11907},
  publisher = {Springer},
  year      = {2020},
  url       = {https://doi.org/10.1007/978-3-030-46147-8},
  doi       = {10.1007/978-3-030-46147-8},
  isbn      = {978-3-030-46146-1},
  timestamp = {Mon, 27 Dec 2021 15:13:42 +0100},
  biburl    = {https://dblp.org/rec/conf/pkdd/2019-2.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
