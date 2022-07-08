# Tank control problem

Master thesis project by Andreas Holck Høeg-Petersen in collaboration with HOFOR.

## Minimizing strategy by maximizing partitions

The strategy in `./GoFastPart_4000_output.json` has 150 leaves. Best result of
100 attempts at converting to a decision tree yields 159 leaves. Applying
`get_boxes()`  with no specification of min or max values reduces the number of
boxes to 86. Converting this to a tree then gives 166 leaves.
