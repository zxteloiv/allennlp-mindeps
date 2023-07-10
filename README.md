The full-fledged AllenNLP is now in maintenance mode.
Please refer to the [repo](https://github.com/allenai/allennlp) for more details and alternatives.

Although there's an AllenNLP-light variant that keeps part of the library work,
it's still relied on a large number of dependencies.

We keep the dependencies here to be minimized.
Only the torch is basically needed.
And the transformers, nltk, scipy are required for some of the metrics.

We remove some of the original functions such as CLI commands, FromParams, Lazy Generics,
data readers and, token embedders, trainers, cached-paths, fairness, and distributed training.
The test cases are also removed due to some of the common dependencies.

## Original Package Overview

<table>
<tr>
    <td><b> allennlp </b></td>
    <td> An open-source NLP research library, built on PyTorch </td>
</tr>
<tr>
    <td><b> allennlp.commands </b></td>
    <td> Functionality for the CLI </td>
</tr>
<tr>
    <td><b> allennlp.common </b></td>
    <td> Utility modules that are used across the library </td>
</tr>
<tr>
    <td><b> allennlp.data </b></td>
    <td> A data processing module for loading datasets and encoding strings as integers for representation in matrices </td>
</tr>
<tr>
    <td><b> allennlp.fairness </b></td>
    <td> A module for bias mitigation and fairness algorithms and metrics </td>
</tr>
<tr>
    <td><b> allennlp.modules </b></td>
    <td> A collection of PyTorch modules for use with text </td>
</tr>
<tr>
    <td><b> allennlp.nn </b></td>
    <td> Tensor utility functions, such as initializers and activation functions </td>
</tr>
<tr>
    <td><b> allennlp.training </b></td>
    <td> Functionality for training models </td>
</tr>
</table>

## Installation

The minimal python version is expected to be greater than 3.11.0.
And the model is tested on pytorch 2.0.1.

Just clone the repo, and install it with `pip install -e .`

