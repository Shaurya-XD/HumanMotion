# Public experimental comparison data

`experimental_model_comparison.csv` is an unchanged copy of the supplied
`references/four_model_comparison.csv`. Unlike the private notebooks and
validation CSV's local training paths, this file contains aggregate metrics only
and is included with the deployable code.

The interface displays test accuracy, test macro F1, and validation macro F1 for
the four five-action HMDB51 Official Split 1 models. Architecture selection was
based on validation macro F1. These figures describe the supplied experiment,
not the current upload or a new evaluation. Historical `forward_latency_ms`
values have no established CPU hosting benchmark context and are deliberately
excluded from the interface. No scores or timings have been generated here.
