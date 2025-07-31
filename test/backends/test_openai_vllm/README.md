
# Test for OpenAI API served by VLLM

## Requirement

anaconda / miniconda / miniforge.

Make sure to run the test with multiple cores available (e.g. in a cloud instance / cluster job).
Although you may think 1 core is enough,
vllm could get stuck due to deadlock if so.

## Installation

Run `./install.sh`

## Testing

``` shell
conda activate mellea
./run_test.sh
```
